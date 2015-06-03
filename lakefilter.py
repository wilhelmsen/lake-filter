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

# Define the logger
LOG = logging.getLogger(__name__)

class LakeFilterException(Exception):
    pass


def get_contour_area(contour, lats, lons):
    new_img = np.zeros((lats.shape[0], lons.shape[0]), dtype=np.uint8)
    cv2.drawContours(new_img, [contour,], -1, 255, -1)
    

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

    LOG.debug("%i %i %i %i"%(lat_min_index, lat_max_index, lon_min_index, lon_max_index))
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
        Given a node at index, get all the children.
        Creates a generator (yield).
        """
        child_index = self.hierarky[index].first_child
        while child_index is not None:
            yield child_index
            child_index = self.hierarky[child_index].next

FILL_VALUE = 255 # np.iinfo(img.dtype).max

class Contour(object):
    def __init__(self, parent, contour_values, global_water_mask, grid_resolution):
        self.parent = parent
        self.children = None  # This will become a list when the children has been filled in.
        self._children_mask = None
        self._full_mask = None
        self.contour_values = contour_values
        self.global_water_mask = global_water_mask
        self._is_land = None
        self.grid_resolution = grid_resolution

    def kill(self):
        print "Killing"

        for c in self.children:
            c = None
        self.children = None
        self._children_mask = None
        self._full_mask = None
        for c in self.contour_values:
            c = None
        self.contour_values = None
        self.global_water_mask = None
        self._is_land = None
        self.grid_resolution = None

        self = None
        gc.collect()
        time.sleep(1)
        print "killed"

    def all_pixels_in_mask_are_the_same(self, mask):
        """
        Checks if all values inside mask is the same on 
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
            self._full_mask = np.zeros_like(self.global_water_mask, dtype=np.uint8)

            # Write the WHOLE contour (including its children).
            cv2.drawContours(self._full_mask, [self.contour_values,], -1, FILL_VALUE, thickness=cv2.cv.CV_FILLED)

            # Convert to boolean mask.
            self._full_mask = self._full_mask == FILL_VALUE # np.where(self._full_mask == FILL_VALUE #, True, False)

            # Make sure that all the pixels in the mask are the same.
            if not self.all_pixels_in_mask_are_the_same(self.mask):
                # Remove the edge pixels.
                self._full_mask = self._remove_edges(self._full_mask)

                # Make sure that the pixels now are the same.
                if not self.all_pixels_in_mask_are_the_same(self.mask):
                    raise LakeFilterException("All pixels inside contour are still not the same!!")

        # Return the full mask.
        return self._full_mask

    @property
    def children_mask(self):
        # No children means emtpy list, not None. If None, it has not yet been set.
        assert(self.children is not None)
        # Cache the children mask.
        if self._children_mask is None:
            # Remark that this is not the full mask property, but the local,
            # private variable.
            # This could be done in another way. We just need something to define the 
            # shape of the children mask, somehow.            
            assert(self._full_mask is not None)
            self._children_mask = np.zeros_like(self._full_mask, dtype=np.bool)

            # If there were any children, thir full masks are added to this children
            # mask.
            for c in self.children:
                self._children_mask = self._children_mask | c.full_mask

        # Return the cashed children mask.
        return self._children_mask

    @property
    def mask(self):
        """
        Returns the water mask for this contour only.

        That means that this full mask without all the full masks from the
        children.
        """
        # No children means emtpy list, not None. If None, it has not yet been set.
        assert(self.full_mask is not None)
        assert(self.children_mask is not None)
        return self.full_mask & ~self.children_mask

    def _remove_edges(self, mask):
        """
        Remove the edges from the contour.
        """
        edge_mask = np.zeros_like(mask, dtype=np.uint8)

        # Draw the contour with 1 pixel as width.
        cv2.drawContours(edge_mask, [self.contour_values,], -1, FILL_VALUE,
                         thickness=1)

        # Convert to true/false mask.
        edge_mask = edge_mask == FILL_VALUE

        if not self.all_pixels_in_mask_are_the_same(edge_mask):
            raise LakeFilterException("All pixels one edge was not the same!")
        
        return mask & ~edge_mask

    def remove_children_full(self, mask):
        """
        Set the children mask.
        """
        assert(self.children is not None)
        if len(self.children) > 0:
            # Remove the children.
            for c in self.children:
                # If the mask is None. The edges has not been removed 
                # from the full mask, if necassary.
                assert(c.mask is not None)
                
                # Remove the full children.
                mask = mask & ~c.full_mask
        return mask


    def load_children(self, node_index, hierarcy, contour_values, global_water_mask):
        self.children = []
        for child_node_index in hierarcy.get_children_indexes(node_index):
            if child_node_index is not None:
                c = Contour(self, contour_values[child_node_index], global_water_mask, self.grid_resolution)
                c.load_children(child_node_index, hierarcy, contour_values, global_water_mask)
                self.children.append(c)
                
    def is_inside(self, contour):
        pt = tuple(self.contour_values[0][0])
        return cv2.pointPolygonTest(contour.contour_values, pt, measureDist=False) > 0
        
    def insert_contour(self, contour):
        # assert(contour.is_inside(self))

        was_not_inside_child = True
        for child in self.children:
            if contour.is_inside(child):
                was_not_inside_child = False
                child.insert_contour(contour)
                break

        if was_not_inside_child:
            # Then it must be a direct child of this.
            contour.parent = self
            self.children.append(contour)

    def get_mask_area(self, lats, lons, threshold=None):
        """
        Calculates the area of a contour, where the 
        children has been removed.

        If threshold is set, it drops out when it reaches the threshold.
        """
        assert(self.children is not None)

        # Remove the children from the mask.
        # Water is 1
        # Land is 0.
        area = 0
        for lat_index, lon_index in np.argwhere(self.mask == True):
            if self.global_water_mask[lat_index, lon_index] == True:
                lat = lats[lat_index]
                # area_of_one_pixel
                area += coordinates.lons_2_km(self.grid_resolution, lat)
                if threshold is not None and area > threshold:
                    return area
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
        # Create a land only mask.
        mask = np.zeros_like(min_lake_area_km2, dtype=np.bool)  # All false.

        # Remove for children.
        for child in self.children:
            mask = mask | child.remove_water_less_than(min_lake_area_km2, lats, lons)

        # Remove current.
        if self.is_water and self.get_mask_area(lats, lons, min_lake_area_km2) >= min_lake_area_km2:
            mask = mask | self.mask

        # Return the resulting mask.
        return mask


class Contours(object):
    def __init__(self, grid_resolution, contours, hierarky, global_water_mask):
        self.grid_resolution = grid_resolution
        self.hierarky = Hierarky(hierarky)
        self.global_water_mask = global_water_mask
        self._top_contours = None
        self.contour_array = contours

    @property
    def top_contours(self):
        if self._top_contours is None:
            # If the top contours have not been set / loaded, 
            # do that here.
            LOG.debug("Structuring the contour objects.")
            self._top_contours = []
            node_idx = self.hierarky.first_root_node_index
            while node_idx is not None:
                c = Contour(None, self.contour_array[node_idx], self.global_water_mask, self.grid_resolution)
                c.load_children(node_idx, self.hierarky, self.contour_array, self.global_water_mask)
                self._top_contours.append(c)
                node_idx = self.hierarky.hierarky[node_idx].next
            # Find top contours that should have been inside an other contour.
            self._find_parents_for_lost_children()
        return self._top_contours

    def _find_parents_for_lost_children(self):
        number_of_top_contours = len(self._top_contours)
        LOG.debug("Finding lost children and inserting them into their parent. Number of top contours: %i"%(number_of_top_contours))
        t = datetime.datetime.now()
        assert(len(self._top_contours) > 0)

        counter = 0
        while counter != len(self._top_contours):
            counter += 1
            assert(len(self._top_contours) > 0)
            c = self._top_contours.pop()
            found = False
            for tc in self._top_contours:
                # print("Comparing", tc, "with", c)
                if c.is_inside(tc):
                    tc.insert_contour(c)
                    found = True
                    counter = 0
                    break

            if not found:
                # Looped through every top contour and none of them were a parent for this contour.
                # This contour is therefore itself a top contour.
                # Insert as first element.
                index = 0
                self._top_contours.insert(index, c)
        LOG.debug("Took: %s seconds."%((datetime.datetime.now()-t).total_seconds()))
        LOG.debug("Done structuring children. Number of top contours: %i (%i)."%(len(self._top_contours), len(self._top_contours) - number_of_top_contours))


if __name__ == "__main__":
    import argparse
    
    def filename(path):
        if not os.path.isfile(path):
            raise argparse.ArgumentTypeError( "'%s' does not exist. Please specify mask file!"%(path))
        return path

    DOMAINS = {
        "fisk": [1.0, 23.3, 2.4, 23.0]
        }
    def domain(domain_name):
        if domain_name not in DOMAINS:
            raise argparse.ArgumentTypeError( "'%s' must be one of '%s'."%(domain_name, "', '".join(DOMAINS)))
        return DOMAINS[domain_name]

    parser = argparse.ArgumentParser(description='Filtering lakes out of the mask.')
    parser.add_argument('fine_land_sea_mask', type=filename, help='The land/sea mask with fine grid.')
    parser.add_argument('min_lake_area_km2', type=float, help='The size (km2) of the lakes to remove from the mask')
    parser.add_argument('--grid-resolution', type=float, help='The size of the output grid, in lat/lons.')

    group = parser.add_mutually_exclusive_group()
    group.add_argument('--domain', type=domain, help="Prints the name of the available buoys for a given data dir.", dest="domain_boundaries")
    group.add_argument('--domain-boundaries', type=float, nargs=4, help="The boundaries of the domain: x0, y0, x1, y1.")
    group.add_argument('--lat-lons', type=float, nargs=4, help="latitude_min, latitude_max, longitude_min, longitude_max.")

    group = parser.add_mutually_exclusive_group()
    group.add_argument('-d', '--debug', action='store_true', help="Output debugging information.")
    group.add_argument('-v', '--verbose', action='store_true', help="Output info.")
    parser.add_argument('--log-filename', type=str, help="File used to output logging information.")
    parser.add_argument('--include-oceans', action='store_true', help="Include oceans in the mask.")
    parser.add_argument('-o', '--output', type=str, help="Output filename.")

    parser.add_argument('--resize-factor', type=float, help='If this is set, the image is resized by this factor.', default=1.0)

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
        LOG.debug("Variables: %s"%(rootgrp.variables))

        # Land/sea mask with distance from land.
        # I.e. everything > 0, is over water.
        LOG.debug("Getting the land/sea mask.")
        if args.lat_lons != None:
            LOG.debug("Limiting the search.")
            lat_min, lat_max, lon_min, lon_max = args.lat_lons

            lat_min_index, lat_max_index, \
                lon_min_index, lon_max_index = get_lat_lon_indexes(rootgrp, lat_min, lat_max, lon_min, lon_max)

            distance_to_land_mask = rootgrp.variables['dst'][lat_min_index:lat_max_index, lon_min_index:lon_max_index]
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
        LOG.debug("Took: %s seconds."%((datetime.datetime.now()-t).total_seconds()))

        # img = cv2.resize(global_water_mask.copy(), (0,0), fx=0.1, fy=0.1)
        # cv2.imshow("Frame", img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        LOG.debug("Setting all values that is not 0 to 255")
        t = datetime.datetime.now()
        # Setting everything that is not land to sea. (Setting everything that is not 0 to 255).
        # 0:   Land.
        # 255: Water.
        global_water_mask = np.array(np.where(global_water_mask == 0, 0, 255), dtype=np.uint8)
        LOG.debug("Took: %s seconds."%((datetime.datetime.now()-t).total_seconds()))

        LOG.debug("Finding contours.")
        t = datetime.datetime.now()
        contours, hierarky = cv2.findContours(global_water_mask.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        LOG.debug("Number of contours: %i. Number of entries in hieracry: %i"%(len(contours), len(hierarky[0])))

        grid_resolution = float(rootgrp.grid_resolution.split("degree")[0])

        LOG.debug("Converting global water mask to boolean.")
        global_water_mask = np.where(global_water_mask == 0, False, True)

        LOG.debug("Loading the contours.")
        
        # All values False.
        new_mask = np.zeros_like(global_water_mask, dtype=np.bool)

        list_of_top_contours = []
        for top_contour in Contours(grid_resolution, contours, hierarky, global_water_mask).top_contours:
            print("Number of children:", len(top_contour.children))
            new_mask |= top_contour.remove_water_less_than(args.min_lake_area_km2, lats, lons)
            if top_contour in list_of_top_contours:
                raise LakeFilterException("KAEMPEPROBLEM!!")
            list_of_top_contours.append(top_contour)
            # top_contour.kill()
        
        image = np.array(np.where(new_mask==True, 255, 0),
                         dtype=np.uint8)

    finally:
        rootgrp.close()


    # Create an image of the global water mask.
    global_water_image = np.array(np.where(global_water_mask==True, 125, 0),
                                  dtype=np.uint8)
    # Add the water that has been removed.
    global_water_image[new_mask & global_water_mask] = 255
        
    if args.resize_factor < 1:
        image = cv2.resize(image,
                           (0,0),
                           fx=args.resize_factor,
                           fy=args.resize_factor)
        
        
        global_water_image = cv2.resize(global_water_image,
                                        (0,0),
                                        fx=args.resize_factor,
                                        fy=args.resize_factor)
        
    cv2.imshow('Global', global_water_image)
    cv2.imshow('Mask', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    LOG.debug("DONE")

