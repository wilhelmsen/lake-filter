#!/usr/bin/env python
# coding: utf-8
import logging
import os
import numpy as np
import cv2
import datetime
import netCDF4
import coordinates
import gc
import time

import multiprocessing as mp

# Define the logger
LOG = logging.getLogger(__name__)


class LakeFilterException(Exception):
    pass


def get_lat_lon_indexes(rootgrp, lat_min, lat_max, lon_min, lon_max):
    """
    Gets the indexes that corresponds to the ranges given in the input.


             lon_min             lon_max
                |                   |
    3 +----+----+----+----+----+----+----+----+
      |    |    |    |    |    |    |    |    |
    2 +----+----+----+----+----+----+----+----+ --- lat_max
      |    |    | x  | x  | x  | x  |    |    |
    1 +----+----+----+----+----+----+----+----+ --- lat_min
      |    |    |    |    |    |    |    |    |
    0 +----+----+----+----+----+----+----+----+
      0    1    2    3    4    5    6    7    8

    Given the indexes above, this should return:
    1, 2, 2, 6

    """
    # The lat variable starts with a great number (around 80) and decreases
    # to around -80)
    # lat_min_index = np.min(np.where(rootgrp.variables['lat'][:] <= lat_min))
    lat_min_index = np.min(np.where(rootgrp.variables['lat'][:] <= lat_max))
    lat_max_index = np.max(np.where(rootgrp.variables['lat'][:] >= lat_min))

    # The lon values starts by around -179 and ends around 179.
    lon_min_index = np.min(np.where(rootgrp.variables['lon'][:] >= lon_min))
    lon_max_index = np.max(np.where(rootgrp.variables['lon'][:] <= lon_max))

    LOG.debug("%i %i %i %i" % (lat_min_index, lat_max_index, lon_min_index,
                               lon_max_index))
    return lat_min_index, lat_max_index, lon_min_index, lon_max_index


class HierarkyNode(object):
    """
    This is only used to put names on the integer values.
    Each node points to another node in the hierarcy.

    -1 means None, so it is here set to None. Each hierarky
    node corresponds to a contour.
    """
    def __init__(self, hierarky_indexes):
        # assert()  # Make sure the type is correct.
        # [Next, Previous, First_Child, Parent]
        self.next, \
            self.prev, \
            self.first_child, \
            self.parent = [None if i == -1 else i
                           for i
                           in hierarky_indexes]


class Hierarky(object):
    """
    Keeping track of the hierarky of the contours.
    """
    def __init__(self, hierarky_from_opencv):
        """
        Initializer for the hierarky.
        """
        self.hierarky = [HierarkyNode(indexes)
                         for indexes
                         in hierarky_from_opencv[0]]

    @property
    def first_root_node_index(self):
        """
        Find the first node with no parent, which is the root
        node.
        """
        for index, node in enumerate(self.hierarky):
            if node.parent is None:
                return index

    @property
    def first_leaf_node(self):
        """
        Find the first node with no children.
        """
        for node in self.hierarky:
            if node.first_child is None:
                return node
        raise LakeFilterException("There was no leaf node... \
Something is very wrong.")

    def get_children_indexes(self, index):
        """
        Given a node at index, get all the children. Creates a
        generator (yield).
        """
        child_index = self.hierarky[index].first_child
        while child_index is not None:
            yield child_index
            child_index = self.hierarky[child_index].next

FILL_VALUE = 255  # np.iinfo(img.dtype).max


class Contour(object):
    def __init__(self, parent, contour_values, global_water_mask,
                 grid_resolution):
        """
        Setting up a contour.
        """
        self.parent = parent
        # This will become a list when the children has been filled in.
        self.children = None
        self._children_mask = None
        self._full_mask = None
        self.contour_values = contour_values
        self.global_water_mask = global_water_mask
        self._is_land = None
        self.grid_resolution = grid_resolution

    def all_pixels_in_mask_are_the_same(self, mask):
        """
        Checks if all the values in the mask has the same value on
        the global water mask.
        """
        assert(self.children is not None)  # Children must be set!

        first_value = None
        for x, y in np.argwhere(mask == True):
            if first_value == None:
                first_value = self.global_water_mask[x, y]

            if self.global_water_mask[x, y] != first_value:
                return False
        return True  # All values inside are the same.

    @property
    def full_mask(self):
        """
        The full mask, holding the whole contour, including children.
        """
        assert(self.children is not None)
        if self._full_mask is None:
            # All values set to 0.
            self._full_mask = np.zeros_like(self.global_water_mask,
                                            dtype=np.uint8)

            # Write the WHOLE contour (including its children).
            cv2.drawContours(self._full_mask, [self.contour_values, ],
                             -1, FILL_VALUE, thickness=cv2.cv.CV_FILLED)

            # Convert to boolean mask. ALL values that True is inside
            # the contour, water or land.
            self._full_mask = self._full_mask == FILL_VALUE

            # The contour may be on the outside of the mask. That is
            # if the contour is water, the contour edge may be on the
            # landside just outside the water.
            #
            # Make sure that all the pixels in the mask are the same,
            # by removing the full mask children from the full mask,
            # and make sure all the inside values are the same.
            if not self.all_pixels_in_mask_are_the_same(self.mask):
                # Remember that self.mask is the current self._full_mask
                # without all the full masks of its children.
                #
                # This is done with the current self._full_mask, as
                # self.full_mask is called in self.mask, but here 
                # self._full_mask is not None and therefore the current
                # version is returned, including the wrong edges.
                #
                # So if every pixel is NOT the same in the mask, the
                # edges are removed below.
                #
                # Remove the edge pixels.
                self._full_mask = self._remove_edges(self._full_mask)

                # Make sure that the pixels now are the same.
                # At this point we know that everything is OK.
                if not self.all_pixels_in_mask_are_the_same(self.mask):
                    raise LakeFilterException("All pixels inside contour \
are still not the same!!")

            # At this point we just want to make sure that there are mask
            # values in the mask. There were some examples where the mask
            # was empty, without reason.
            # At least one pixel must be True.
            assert(np.any(self._full_mask))

        # Return the full mask.
        return self._full_mask

    @property
    def children_mask(self):
        """
        A mask that contains all the children full masks.
        """
        # No children means emtpy list, not None. If None, it has not
        # yet been set.
        assert(self.children is not None)
        # Cache the children mask.
        if self._children_mask is None:
            # Remark that this is not the full mask property, but the
            # local, private variable.
            # This could be done in another way. We just need something
            # to define the shape of the children mask, somehow.
            assert(self._full_mask is not None)
            self._children_mask = np.zeros_like(self._full_mask, dtype=np.bool)

            # If there were any children, thir full masks are added to this
            # children mask.
            for c in self.children:
                self._children_mask = self._children_mask | c.full_mask

        # Return the cashed children mask.
        return self._children_mask

    @property
    def mask(self):
        """
        Returns the mask for this contour only, without its children.

        I.e. the full mask without all the children full masks.
        """
        # No children means emtpy list, not None. If None, it has not yet
        # been set.
        assert(self.full_mask is not None)
        assert(self.children_mask is not None)
        return self.full_mask & ~self.children_mask

    def _remove_edges(self, mask):
        """
        Remove the edges from the contour.

        This is e.g. used if the contour is set on the "wrong" side of the
        mask. If the inside of the mask is water and the contour is set on
        the land surrounding it. By removing the land, everything inside the
        mask is of the same type.
        """
        edge_mask = np.zeros_like(mask, dtype=np.uint8)

        # Draw the contour with 1 pixel as width.
        cv2.drawContours(edge_mask, [self.contour_values, ], -1, FILL_VALUE,
                         thickness=1)

        # Convert to true/false mask.
        edge_mask = edge_mask == FILL_VALUE

        if not self.all_pixels_in_mask_are_the_same(edge_mask):
            raise LakeFilterException("All pixels one edge was not the same!")

        return mask & ~edge_mask

    def build_contour_tree(self, node_index, hierarcy, contour_values,
                           global_water_mask):
        """
        Recursively build the contour tree. I.e. recursively setting up
        contour objects for every child in the hierarchy.
        """
        self.children = []
        for child_node_index in hierarcy.get_children_indexes(node_index):
            if child_node_index is not None:
                c = Contour(self, contour_values[child_node_index],
                            global_water_mask, self.grid_resolution)
                c.build_contour_tree(child_node_index, hierarcy,
                                     contour_values, global_water_mask)
                self.children.append(c)

    def is_inside(self, contour):
        """
        Checks if the input contour is inside the current contour (self).
        """
        # Pick any point of the contour values.
        point_to_check = tuple(self.contour_values[0][0])

        # Check if the point is inside the contour.
        return cv2.pointPolygonTest(contour.contour_values, point_to_check,
                                    measureDist=False) > 0

    def insert_contour(self, contour):
        """
        Trying to insert the given contour into this contour (self).

        If it is inside the contour it tries to insert it into its children.

        If this does not succede, the contour must be one of the direct
        children of this contour (self). But the children of this contour 
        (self) should may be have been inside the current contour (contour).
        It therefore first tries to insert the children (self.children) into 
        the current contour (contour). After, the contour (cntour) is inserted
        into the this contour (self).

        """
        t = datetime.datetime.now()
        if not contour.is_inside(self):
            # LOG.debug("Was not inside took %0.3f"%((datetime.datetime.now()-t).total_seconds()))
            return False
        LOG.debug("Was inside took %0.3f"%((datetime.datetime.now()-t).total_seconds()))

        # At this point we know that the contour should be inside this
        # contour (self) somewhere.

        # First try the children.
        was_inserted_into_one_of_the_children = False
        for child in self.children:
            t = datetime.datetime.now()
            if child.insert_contour(contour):
                LOG.debug("Inserting contour into child took %0.3f"%((datetime.datetime.now()-t).total_seconds()))
                # The contour cannot be inside several children.
                return True


        # If the contour (contour) was NOT inserted into one of the children,
        # it is a child of this contour (self).
        #
        # First, let us make sure the children contours of this contour 
        # (self) chould not be inserted into the contour (contour).
        self_children = []
        while len(self.children) > 0:
            # Use the pop functionality to avoid removing list items
            # in an "unhealthy" way.
            child = self.children.pop()

            t = datetime.datetime.now()
            # Insert the child into the contour (contour).
            if not contour.insert_contour(child):
                LOG.debug("Inserting child into contour took %0.3f"%((datetime.datetime.now()-t).total_seconds()))
                # It is still a child of this contour (self).
                self_children.append(child)

        # The children has been inserted, if needed.
        # Now set the record right!
        contour.parent = self
        self_children.append(contour)
        self.children = self_children

        # The contour was inserted right there...  --->   x
        # GREAT SUCCESS!!!
        return True
        
    def get_mask_area(self, lats, lons, threshold=None):
        """
        Calculates the area of a contour, where the children has
        been removed.

        If threshold is set, it drops out when it reaches the threshold.
        The area is then set to the threshold.
        """
        assert(self.children is not None)

        # Remove the children from the mask.
        # Water is 1
        # Land is 0.
        area = 0
        for lat_index, lon_index in np.argwhere(self.mask == True):
            lat = lats[lat_index]

            # Add the area of one pixel.
            area += coordinates.lons_2_km(self.grid_resolution, lat) * \
                coordinates.lats_2_km(self.grid_resolution)

            # If the nature of the request requires it to be below some
            # threshold, there is no need to calculate the whole area.
            # Jump out from here.
            if threshold is not None and area > threshold:
                return threshold
        return area

    @property
    def is_water(self):
        """
        Checks if the contour is water.
        All the contours inside the mask is of the same type. That means
        that if any point in the mask is water, then all values are water.
        """
        # Pick the first pixel in the contour (where the children has been
        # removed.
        # show_masks([self.global_water_mask, self.mask])
        if not np.any(self.mask):
            LOG.warning("""An empty mask... Why is it then a mask at all?""")
            raise LakeFilterException("The mask is empty...")
        else:
            lat_index, lon_index = np.argwhere(self.mask == True)[0]
            

        # Check if that pixel is water or not.
        return self.global_water_mask[lat_index, lon_index]

    def has_parent(self):
        """
        Checks if contour has parents. I.e. if it is a top contour or not.
        """
        return self.parent is not None

    def has_children(self):
        """
        Checks if the contour has children.
        """
        return len(self.children) > 0

    def remove_water_less_than(self, min_lake_area_km2, lats, lons):
        """
        Removing all water that is less than threshold inside the current
        contour.
        """
        LOG.debug("Number of contours from here: %i"%(self.count()))
        
        # Create a land only mask.
        mask = np.zeros_like(min_lake_area_km2, dtype=np.bool)  # All false.

        # Remove for children.
        for child in self.children:
            mask = mask | child.remove_water_less_than(min_lake_area_km2,
                                                       lats, lons)

        # Remove current.
        if self.is_water and self.get_mask_area(lats,
                                                lons,
                                                min_lake_area_km2) \
                                                >= min_lake_area_km2:
            mask = mask | self.mask

        # Return the resulting mask.
        return mask

    def count(self):
        """
        Counts the number of contours inside a contour, including
        it self.
        """
        c = 1  # Including one self.
        for child in self.children:
            c += child.count()
        return c


class Contours(object):
    def __init__(self, grid_resolution, contours, hierarky, global_water_mask):
        self.grid_resolution = grid_resolution
        self.hierarky = Hierarky(hierarky)
        self.global_water_mask = global_water_mask
        self._top_contours = None
        self.contour_array = contours

    @property
    def top_contours(self):
        """
        The top contours from this hierarky.
        If not set, they are calculated and organised.

        Some of the contours are not structured correctly, in the
        hierarky from opencv2. This is taken care of at the bottom:
        find_parents_for_lost_children.
        """
        if self._top_contours is None:
            # If the top contours have not been set / loaded,
            # do that here.
            LOG.info("Structuring the contour objects.")
            self._top_contours = []

            # Get the first root note from the hierarky as starting point.
            node_idx = self.hierarky.first_root_node_index

            # Go through all the top contours and insert them where they
            # belong.
            while node_idx is not None:
                # Create a top contour.
                c = Contour(None, self.contour_array[node_idx],
                            self.global_water_mask, self.grid_resolution)

                # Insert the top contour children.
                c.build_contour_tree(node_idx,
                                     self.hierarky,
                                     self.contour_array,
                                     self.global_water_mask)

                # If this new contour is inside one of the top contours
                # it should be inserted there.
                inserted = False
                for tc in self._top_contours:
                    if tc.insert_contour(c):
                        inserted = True
                        break
                    elif c.insert_contour(tc):
                        self._top_contours.remove(tc)
                        break

                # If the contour was not inserted into one of the top contours
                # it is itself a top contour. It may exist inside one of the
                # contours that is beeing built later, but that is handled in
                # find_parents_for_lost_children below.
                if not inserted:
                    # Add the top contour to the list of top contours.
                    self._top_contours.append(c)

                # Go to the next index in the hierarky.
                node_idx = self.hierarky.hierarky[node_idx].next

            LOG.info("Done creating contours.")

            # Not all the contours are inserted correctly in the openCV call.
            # Find top contours that should have been inside an other contour
            # and insert them where they belong.
            # I.e. organise the contours properly.
            LOG.info("Organizing lost children.")
            t = datetime.datetime.now()
            self._find_parents_for_lost_children()
            print (datetime.datetime.now() - t).total_seconds()
        return self._top_contours

    def _find_parents_for_lost_children(self):
        """
        When the contours are made by openCV, some contours are not inserted
        into its parent contour, but is a contour with no parent, lost
        children.

        This function loops through all the top contours, and checks if one of
        them should have been inside one of the other top contours.

        If so, it is inserted in that contour and is no longer a top contour.
        """
        number_of_top_contours = len(self._top_contours)
        LOG.debug("Finding lost children and inserting them into their parent. \
Number of top contours: %i" % (number_of_top_contours))
        t = datetime.datetime.now()
        assert(number_of_top_contours > 0)

        counter = 0
        total_counter = 0
        while counter != len(self._top_contours):
            counter += 1
            total_counter += 1

            assert(len(self._top_contours) > 0)
            c = self._top_contours.pop()

            inserted = False
            for tc in self._top_contours:
                if tc.insert_contour(c):
                    inserted = True
                    counter = 0
                    break

            if not inserted:
                # Looped through every top contour and none of them were a
                # parent for this contour. This contour is therefore itself
                # a top contour. Insert as first element.
                index = 0
                self._top_contours.insert(index, c)
        LOG.info("Total cycles: %i. Took: %s seconds." % (
                total_counter,
                (datetime.datetime.now() - t).total_seconds()))
        LOG.info("Done structuring children. Number of top contours: %i (%i)."
                  % (len(self._top_contours),
                     len(self._top_contours) - number_of_top_contours))


def remove_water(top_contour, min_lake_area_km2, lats, lons, output_queue):
    LOG.debug("Removing water...")
    output_queue.put(top_contour.remove_water_less_than(min_lake_area_km2, lats, lons))


def remove_water_in_new_process(i, top_contour, min_lake_area_km2, lats, lons, output):
    process = mp.Process(target=remove_water,
                         args = (top_contour, min_lake_area_km2, lats, lons, output)
                         )
    #p = mp.Process(target=remove_water,
    #               args=(top_contour, min_lake_area_km2, lats, lons, output)
    #               )
    LOG.debug("Starting process %i."%(i))
    process.start()
    LOG.debug("Finished process %i."%(i))
    

def do_it(top_contour, min_lake_area_km2, lats, lons, i):# , output_queue):
    """
    Remove the water.
    """
    # output_queue.put(top_contour.remove_water_less_than(min_lake_area_km2, lats, lons))
    LOG.info("Doing %03i."%(i))
    mask = remove_water_in_new_process(top_contour, min_lake_area_km2, lats, lons)
    # mask = top_contour.remove_water_less_than(min_lake_area_km2, lats, lons)
    LOG.info("%03i done."%(i))
    return mask

def show_masks(masks):
    """
    Showing the masks.
    """
    index = 0
    for mask in masks:
        index += 1
        image = np.array(np.where(mask==True, 255, 0), dtype=np.uint8)
        factor = 1000.0/max(image.shape)
        factor = min(1.0, factor)
        if factor < 1:
            image = cv2.resize(image, (0, 0), fx=factor, fy=factor)

        filename = "/home/hw/Desktop/mask_%i.png"%(index)
        cv2.imwrite(filename, image)
        print "File written to %s"%(filename)

        # cv2.imshow('Mask %i' % (index), image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()


if __name__ == "__main__":
    import argparse

    def filename(path):
        if not os.path.isfile(path):
            raise argparse.ArgumentTypeError("'%s' does not exist. Please \
specify mask file!" % (path))
        return path

    DOMAINS = {
        "fisk": [1.0, 23.3, 2.4, 23.0]
        }

    def domain(domain_name):
        if domain_name not in DOMAINS:
            raise argparse.ArgumentTypeError("'%s' must be one of '%s'." %
                                             (domain_name,
                                              "', '".join(DOMAINS)))
        return DOMAINS[domain_name]

    parser = argparse.ArgumentParser(
        description='Filtering lakes out of the mask.')
    parser.add_argument('fine_land_sea_mask', type=filename,
                        help='The land/sea mask with fine grid.')
    parser.add_argument('min_lake_area_km2', type=float,
                        help='The size (km2) of the lakes to remove')
    parser.add_argument('--grid-resolution', type=float,
                        help='The size of the output grid, in lat/lons.')

    group = parser.add_mutually_exclusive_group()
    group.add_argument('--domain', type=domain,
                       help="The domain to create the mask for.",
                       dest="domain_boundaries")
    group.add_argument('--domain-boundaries', type=float, nargs=4,
                       help="The boundaries of the domain: x0, y0, x1, y1.")
    group.add_argument('--lat-lons', type=float, nargs=4,
                       help="lat_min, lat_max, lon_min, lon_max.")

    group = parser.add_mutually_exclusive_group()
    group.add_argument('-d', '--debug', action='store_true',
                       help="Output debugging information.")
    group.add_argument('-v', '--verbose', action='store_true',
                       help="Output info.")
    parser.add_argument('--log-filename', type=str,
                        help="File used to output logging information.")
    parser.add_argument('--include-oceans', action='store_true',
                        help="Include oceans in the mask.")
    parser.add_argument('-o', '--output', type=str,
                        help="Output filename.")

    parser.add_argument('--resize-factor', type=float,
                        help="If set, the output image is reduced by this.",
                        default=1.0)

    # Do the parser.
    args = parser.parse_args()

    # Set the log options.
    if args.debug:
        logging.basicConfig(filename=args.log_filename, level=logging.DEBUG)
    elif args.verbose:
        logging.basicConfig(filename=args.log_filename, level=logging.INFO)
    else:
        logging.basicConfig(filename=args.log_filename, level=logging.WARNING)

    # Output what is in the args variable.
    LOG.debug(args)

    rootgrp = netCDF4.Dataset(args.fine_land_sea_mask, 'r')

    try:
        LOG.debug("Variables: %s" % (rootgrp.variables))

        # Land/sea mask with distance from land.
        # I.e. everything > 0, is over water.
        LOG.debug("Getting the land/sea mask.")
        if args.lat_lons != None:
            LOG.debug("Limiting the search.")
            lat_min, lat_max, lon_min, lon_max = args.lat_lons

            lat_min_index, \
                lat_max_index, \
                lon_min_index, \
                lon_max_index = get_lat_lon_indexes(rootgrp, lat_min,
                                                    lat_max, lon_min, lon_max)

            distance_to_land_mask = rootgrp.variables['dst'][
                lat_min_index:lat_max_index, lon_min_index:lon_max_index]
            lats = rootgrp.variables['lat'][lat_min_index:lat_max_index]
            lons = rootgrp.variables['lon'][lon_min_index:lon_max_index]
        else:
            LOG.debug("Getting all values..")
            distance_to_land_mask = rootgrp.variables['dst'][:, :]
            lats = rootgrp.variables['lat'][:]
            lons = rootgrp.variables['lon'][:]

        LOG.debug("Converting land sea mask to np.array.")
        t = datetime.datetime.now()
        global_water_mask = np.array(distance_to_land_mask, dtype=np.uint8)
        LOG.debug("Took: %s seconds." % (
                (datetime.datetime.now() - t).total_seconds()))

        # img = cv2.resize(global_water_mask.copy(), (0,0), fx=0.1, fy=0.1)
        # cv2.imshow("Frame", img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        LOG.debug("Setting all values that is not 0 to 255")
        t = datetime.datetime.now()
        # Setting everything that is not land to sea. (Setting everything that
        # is not 0 to 255).
        # 0:   Land.
        # 255: Water.
        global_water_mask = np.array(np.where(global_water_mask == 0, 0, 255),
                                     dtype=np.uint8)
        LOG.debug("Took: %s seconds." % (
                (datetime.datetime.now() - t).total_seconds()))

        # Finding the contours with opencv2. The RETR_CCOMP method returns 
        # a tree structure of the contours. The structure is saved in 
        # the hierarky variable.
        #
        # Not all contours are inserted inside the correct contour. The top
        # contours are therefore tried inserted into the correct contour
        # when the top contours tree is created.
        LOG.debug("Finding contours.")
        t = datetime.datetime.now()
        contours, hierarky = cv2.findContours(global_water_mask.copy(),
                                              cv2.RETR_CCOMP,
                                              cv2.CHAIN_APPROX_NONE)
        LOG.debug("Number of contours: %i. Number of entries in hieracry: %i"
                  % (len(contours), len(hierarky[0])))

        
        # This is the grid resolution of the grids in the input land sea mask
        # netcdf file.
        grid_resolution = float(rootgrp.grid_resolution.split("degree")[0])

        # In order to use the mask from the file, the values must be converted.
        LOG.debug("Converting global water mask to boolean.")
        # global_water_mask = np.where(global_water_mask == 0, False, True)
        global_water_mask = global_water_mask != 0


        # List of contours holds all the contours. It makes sure the same
        # contour is not checked twice.
        top_contours = Contours(grid_resolution, contours, hierarky,
                                global_water_mask).top_contours

        # Count the number of contours. No contours should disappear in the
        # process above.
        total_number_of_contours = 0
        for top_contour in top_contours:
            total_number_of_contours += top_contour.count()
        LOG.info("Total number of contours %i. Was: %i."%(total_number_of_contours, len(hierarky[0])))

        # Make sure the number of contours are the same now as
        # rigth after they were found.
        if total_number_of_contours != len(hierarky[0]):
            raise LakeFilterException("The number of inserted contours \
does not match the number of contours found by opencv2.")

        # For diagnostics·
        list_of_top_contours = []
        total_time_s = 0
        total_countours_done = 0
        counter = 0

        LOG.debug("Loading the contours.")
        mask_with_removed_lakes = np.zeros_like(global_water_mask, dtype=np.bool)

        cpu_count = mp.cpu_count()
        number_of_processes = min(len(top_contours), cpu_count)

        LOG.debug("Number of cpus: %i"%(cpu_count))
        outputs = mp.Queue()

        for i in range(number_of_processes):
            top_contour = top_contours.pop()
            remove_water_in_new_process(i, top_contour, args.min_lake_area_km2, lats, lons, outputs)

        while len(top_contours) > 0:
            # At this poit number_of_processes should be put into the output queue.
            # 
            mask = outputs.get()
            i += 1
            mask_with_removed_lakes |= mask
            top_contour = top_contours.pop()
            remove_water_in_new_process(i, top_contour, args.min_lake_area_km2, lats, lons, outputs)
        
        for j in range(number_of_processes):
            i += 1
            LOG.debug("Waiting for output.")
            mask = outputs.get()
            mask_with_removed_lakes |= mask

        LOG.info("Number of lakes removed: %i. Total number of contours: %i."%(i, total_number_of_contours))

        """
        while len(top_contours) > 0:
            top_contour = top_contours.pop()
            
            # Waits forever untill one is available.
            mask = outputs.get()
            remove_water_in_new_process(top_contour, min_lake_area_km2, lats, lons, outputs)
            mask_with_removed_lakes |= mask
            


        pool = mp.Pool()
        # for top_contour in tp = top_contours
        LOG.info("Starting processing...")
        for mask in [pool.apply_async(do_it, args = (top_contour, args.min_lake_area_km2, lats, lons, i)) for i, top_contour in enumerate(top_contours)]:
            LOG.info("Appending child mask...")
            mask_with_removed_lakes |= mask.get()  #mask
        pool.close()
        pool.join()
        print "Joined!"
        # for mask in masks:
        #     mask_with_removed_lakes |= mask  #mask
        
"""
        """
        # For every top contour remove water smaller than threshold.
        for top_contour in top_contours:
            number_of_contours_in_top_contour = top_contour.count()
            LOG.debug("Total number of contours in top contour: %i" % (number_of_contours_in_top_contour))

            counter += 1
            t = datetime.datetime.now()
            LOG.debug("Top contour no.: %i of %i." % (counter, len(top_contours)))


            # DO IT!!
            # Starting it all in a new process. This is mostly done because
            # (we think) the garbage collector is a bit slow...
            output = mp.Queue()
            p = mp.Process(target=do_it,
                           args=(top_contour, args.min_lake_area_km2,
                                 lats, lons, output))
            p.start()
            mask_with_removed_lakes |= output.get()
            p.join()

            # Diagnostics.
            process_time_s = (datetime.datetime.now() - t).total_seconds()
            total_countours_done += number_of_contours_in_top_contour
            total_time_s += process_time_s

            contours_left = total_number_of_contours - total_countours_done
            avg_time_each_s = total_time_s / total_countours_done

            avg_time_left_s = contours_left * avg_time_each_s
            LOG.info("Expected time left %0.3fs. Expected end time: %s" % (avg_time_left_s, datetime.datetime.now() + datetime.timedelta(seconds=avg_time_left_s)))

            LOG.debug("Getting resulting mask for top contour took %0.3f seconds"%(process_time_s))
            t = datetime.datetime.now()

            # Make sure the contour has not been processed twice.
            if top_contour in list_of_top_contours:
                raise LakeFilterException("HUGE PROBLEM!!!!")
            list_of_top_contours.append(top_contour)
            LOG.debug("Checking took %0.3f seconds"%((datetime.datetime.now() - t).total_seconds()))
            """
    finally:
        rootgrp.close()

    """
    image = np.array(np.where(new_mask == True, 255, 0),
                     dtype=np.uint8)

    # Create an image of the global water mask.
    global_water_image = np.array(np.where(global_water_mask == True, 125, 0),
                                  dtype=np.uint8)
    # Add the water that has been removed.
    global_water_image[new_mask & global_water_mask] = 255

    # Create an image of the global water mask.
    global_water_image = np.array(np.where(global_water_mask==True, 125, 0),
    dtype=np.uint8)

    if args.resize_factor < 1:
    global_water_image = cv2.resize(global_water_image,
    (0,0),
    fx=args.resize_factor,
    fy=args.resize_factor)

    cv2.imshow('Global', global_water_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    """
    LOG.debug("Showing result.")
    show_masks([global_water_mask, mask_with_removed_lakes])

    LOG.debug("DONE")
