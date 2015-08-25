#!/usr/bin/env python
# coding: utf-8
import logging
import os
import numpy as np
import cv2
import datetime
import netCDF4
import gc
import time
import multiprocessing as mp

import decorators
import domain

__doc__ = """Lake filter.

Removes water (lakes) of a certain size, based on the input map.

Usage:
  {filename} <fine_land_sea_mask> <min_lake_area_km2> (--resolution <res> | (--dlat <dlat> --dlon <dlon>)) [--domain <domain> | (--lats <south> <north> --lons <west> <east>)] [-d|-v] [--log-filename=<filename>]
  {filename} (-h | --hep)

Positional arguments:
  fine_land_sea_mask    The land/sea mask with fine grid.
  min_lake_area_km2     The size (km2) of the lakes to remove

Options:
  -h, --help                       Show this help message and exit.
  --domain=<domain>                The domain to create the mask for.
                                   Available domains: ["{domains}"].
  --lats                           Latitude boundaries. Min value and max value.
  --lons                           Longitude boundaries. Min value and max value.
  --resolution=<res>               Output resolution. Equals: dlat = dlon = resolution.
  --dlat=<dlat>                    Delta latitude. Latitude steps.
  --dlon=<dlon>                    Delta longitude. Longitude steps.
  -d --debug                       Output debugging information.
  -v --verbose                     Output info.
  --log-filename=<filename>        File used to output logging information.
""".format(domains="\", \"".join(domain.DOMAINS.keys()), filename=__file__)


class LakeFilterException(Exception):
    pass


# Define the logger
LOG = logging.getLogger(__name__)
FILL_VALUE = 255  # np.iinfo(img.dtype).max

# Coordinates.
EARTH_MEAN_RADIUS_KM = 6371
EARTH_MEAN_DIAMETER_KM = 2.0 * np.pi * EARTH_MEAN_RADIUS_KM
EARTH_ONE_MEAN_DEG_KM = EARTH_MEAN_DIAMETER_KM / 360.0


def length_of_one_mean_degree_at_latitude_km(latitude):
    return EARTH_ONE_MEAN_DEG_KM * np.cos(np.deg2rad(latitude))


def lons_2_km(longitudes, latitude):
    return longitudes * length_of_one_mean_degree_at_latitude_km(latitude)


def lats_2_km(latitudes):
    return latitudes * EARTH_ONE_MEAN_DEG_KM


def km_2_lons(distance_km, latitude):
    return distance_km / length_of_one_mean_degree_at_latitude_km(latitude)


def km_2_lats(distance_km):
    return distance_km / EARTH_ONE_MEAN_DEG_KM


def get_max_min_indexes(array, min_value, max_value):
    """
    Gets indexes where the array is
    """
    # Just make sure that the minimum value actually is the minimumvalue and
    # that the maximum value actually si the maximum value.
    min_value, max_value = min(min_value, max_value), max(min_value, max_value)

    LOG.debug("Got range index. min_value: %f. max_value: %f" % (min_value,
                                                                 max_value))

    # Get the idexes.
    indexes = np.where(
        (array >= min_value) &
        (array <= max_value)
        )

    # If no values found...
    if indexes.size == 0:
        return None, None

    return np.min(indexes), np.max(indexes)


def get_lat_lon_indexes(orig_lats, orig_lons, lat_min, lat_max,
                        lon_min, lon_max, dlat, dlon):
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


    dlat and dlon are there to add the 
    """
    if lat_min is not None and lat_max is not None:
        # The lat variable starts with a great number (around 80) and decreases
        # to around -80)
        # lat_min_index = np.min(np.where(rootgrp.variables['lat'][:] <= lat_min))
        lat_min, lat_max = min(lat_min, lat_max), max(lat_min, lat_max)
        lat_min_index, lat_max_index = get_max_min_indexes(orig_lats,
                                                           lat_min - dlat / 2.0,
                                                           lat_max + dlat / 2.0)
    else:
        # Use all values.
        lat_min_index = 0
        lat_max_index = int(orig_lats.shape[0]) - 1

    if lon_min is not None and lon_max is not None:
        # The lon variable starts by highest number. High number => low index.
        lon_min, lon_max = min(lon_min, lon_max), max(lon_min, lon_max)
        lon_min_index, lon_max_index = get_max_min_indexes(orig_lons,
                                                           lon_min - dlat / 2.0,
                                                           lon_max + dlat / 2.0)
    else:
        lon_min_index = 0
        lon_max_index = int(orig_lons.shape[0]) - 1  # -1 because we start at 0...

    LOG.debug("Lat, lon indexes: (%i, %i, %i, %i)." % (lat_min_index, lat_max_index,
                                                       lon_min_index, lon_max_index))
    return lat_min_index, lat_max_index, lon_min_index, lon_max_index


class HierarkyNode(object):
    """
    This is only used to put names on the integer values.
    Each node points to another node in the hierarcy, by its
    index in the hierarky variable that comes out of opencv2,
    find contours.

    -1 means that there is no value, so it is here set to None.
    Each hierarky node corresponds to a contour.
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


class Contour(object):
    def __init__(self, parent, contour_values, bool_water_mask):
        """
        Setting up a contour.
        """
        self.parent = parent
        # This will become a list when the children has been filled in.
        self.children = None
        self._children_mask = None
        self._full_mask = None
        self.contour_values = contour_values
        self.bool_water_mask = bool_water_mask
        self._is_land = None

    def all_pixels_in_mask_are_the_same(self, mask):
        """
        Checks if all the values in the mask has the same value on
        the global water mask.
        """
        assert(self.children is not None)  # Children must be set!

        first_value = None
        for x, y in np.argwhere(mask == True):
            if first_value == None:
                first_value = self.bool_water_mask[x, y]

            if self.bool_water_mask[x, y] != first_value:
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
            self._full_mask = np.zeros_like(self.bool_water_mask,
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
                           bool_water_mask):
        """
        Recursively build the contour tree. I.e. recursively setting up
        contour objects for every child in the hierarchy.
        """
        self.children = []
        for child_node_index in hierarcy.get_children_indexes(node_index):
            if child_node_index is not None:
                c = Contour(self, contour_values[child_node_index],
                            bool_water_mask)
                c.build_contour_tree(child_node_index, hierarcy,
                                     contour_values, bool_water_mask)
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

    @decorators.timer
    def insert_contour(self, contour):
        """
        Trying to insert the given contour into this contour (self).

        If it is inside the contour it tries to insert it into its children.

        If this does not succede, the contour must be one of the direct
        children of this contour (self). But the children of this contour
        (self) should may be have been inside the current contour (contour)
        It therefore first tries to insert the children (self.children) into
        the current contour (contour). After, the contour (cntour) is inserted
        into the this contour (self).

        """
        if not contour.is_inside(self):
            return False

        # At this point we know that the contour should be inside this
        # contour (self) somewhere.
        #
        # First try the children.
        was_inserted_into_one_of_the_children = False
        for child in self.children:
            if child.insert_contour(contour):
                LOG.debug("Inserting contour into child")
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

            # Insert the child into the contour (contour).
            if not contour.insert_contour(child):
                LOG.debug("Inserting child into contour")
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

    @decorators.timer
    def get_mask_area(self, lats, lons, orig_grid_resolution, threshold=None):
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
            area += lons_2_km(orig_grid_resolution, lat) * \
                lats_2_km(orig_grid_resolution)

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
        if not np.any(self.mask):
            LOG.warning("""An empty mask... Why is it then a mask at all?""")
            raise LakeFilterException("The mask is empty...")
        else:
            lat_index, lon_index = np.argwhere(self.mask == True)[0]

        # Check if that pixel is water or not.
        return self.bool_water_mask[lat_index, lon_index]

    @decorators.timer
    def has_parent(self):
        """
        Checks if contour has parents. I.e. if it is a top contour or not.
        """
        return self.parent is not None

    @decorators.timer
    def has_children(self):
        """
        Checks if the contour has children.
        """
        return len(self.children) > 0

    @decorators.timer
    def remove_water_less_than(self, lats, lons, orig_grid_resolution, min_lake_area_km2):
        """
        Removing all water that is less than threshold inside the current
        contour.
        """
        LOG.debug("Number of contours from here: %i" % (self.count()))

        # Create a land only mask.
        mask = np.zeros_like(min_lake_area_km2, dtype=np.bool)  # All false.

        # Remove for children.
        for child in self.children:
            mask = mask | child.remove_water_less_than(lats, lons, orig_grid_resolution,
                                                       min_lake_area_km2)

        # Remove current.
        if self.is_water and self.get_mask_area(lats,
                                                lons,
                                                orig_grid_resolution,
                                                min_lake_area_km2) \
                                                >= min_lake_area_km2:
            mask = mask | self.mask

        # Return the resulting mask.
        return mask

    @decorators.timer
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
    @decorators.timer
    def __init__(self, contours, hierarky, bool_water_mask):
        self.hierarky = Hierarky(hierarky)
        self.bool_water_mask = bool_water_mask
        self._top_contours = None
        self.contour_array = contours

    @property
    @decorators.timer
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
                            self.bool_water_mask)

                # Insert the top contour children.
                c.build_contour_tree(node_idx,
                                     self.hierarky,
                                     self.contour_array,
                                     self.bool_water_mask)

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
            self._find_parents_for_lost_children()

        assert(len(self.hierarky.hierarky) == sum([tc.count() for tc in self._top_contours]))
        return self._top_contours

    @staticmethod
    @decorators.timer
    def get_top_contours(water_mask_bool):
        """
        Getting the contours from the intput water mask. The values in the
        input mask may only be True or False, where True means water
        and False means land. Therefore water mask.
        """
        LOG.debug("Finding contours.")

        # Finding the contours with opencv2. The RETR_CCOMP method returns
        # a tree structure of the contours. The structure is saved in
        # the hierarky variable.
        #
        # Not all contours are inserted inside the correct contour. The top
        # contours are therefore tried inserted into the correct contour
        # when the top contours tree is created.
        #
        # First convert the water mask to an int mask, with values 255 for
        # water and 0 for land.
        water_mask_int8 = bool_water_mask_2_int_water_mask(water_mask_bool)

        # Find contours changes the input mask. Therefore a copy.
        contours, hierarky = cv2.findContours(water_mask_int8.copy(),
                                              cv2.RETR_CCOMP,
                                              cv2.CHAIN_APPROX_NONE)
        top_contours = []
        if len(contours) > 0:
            # If there actually were any contours.
            LOG.debug("Number of contours: %i. Number of entries in hieracry: %i"
                      % (len(contours), len(hierarky[0])))

            # top_contours holds all the contours in a correct hierarky.
            top_contours = Contours(contours, hierarky,
                                    water_mask_bool).top_contours

        # Return the list of top contours. If no contours were found, an empty
        # list is returned. Else the top contours of the water mask is returned.
        return top_contours

    @decorators.timer
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

        # If the following fails, there are no contours and something is wrong.
        assert(number_of_top_contours > 0)

        # Loop through every element in the list of top contours and try to
        # insert them into the other contours, one by one.
        # This is done until none of the contours could be inserted any more.
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
        LOG.debug("Total cycles: %i." % (total_counter))
        LOG.debug("Done structuring children. Number of top contours: %i (%i)."
                  % (len(self._top_contours),
                     len(self._top_contours) - number_of_top_contours))


@decorators.timer
def int_water_mask_2_bool_water_mask(int_water_mask):
    """
    Converts an int mask to a boolean mask. I.e. every value that is zero is land.
    all other values are water, also the ones that are masked.
    """
    # 0:   Land.
    # Else: Water.
    # Masked: Water.
    #
    # return int_water_mask != 0
    # return ~(int_water_mask == 0)
    return np.array(np.where(int_water_mask != 0, True, False), dtype=np.bool) | \
        np.array(np.where(int_water_mask.mask == True, True, False), dtype=np.bool)


@decorators.timer
def bool_water_mask_2_int_water_mask(bool_water_mask):
    """
    Converts a boolean mask to int8 mask. I.e. if the boolean mask is True
    (water) the value is set to 255, else 0.

    This is used to find the contours and also to print.
    """
    return np.array(np.where(bool_water_mask == True, 255, 0), dtype=np.uint8)


@decorators.timer
def get_slice_indexes(lat_start, lat_stop, lon_start, lon_stop,
                      slice_step=400, overlap=50):
    assert(slice_step > overlap * 2)
    lat_pointer = lat_start
    lon_pointer = lon_start
    while lon_pointer < lon_stop:
        while lat_pointer < lat_stop:
            yield (lat_pointer, lat_pointer + slice_step), (lon_pointer, lon_pointer + slice_step)
            lat_pointer += slice_step - overlap
        lat_pointer = lat_start
        lon_pointer += slice_step - overlap


def show_masks(masks):
    """
    Showing the masks.
    """
    index = 0
    for mask in masks:
        index += 1
        image = np.array(np.where(mask == True, 255, 0), dtype=np.uint8)
        #if np.all(image):
        #    raise LakeFilterException("All values True. All water.")

        max_pixels = 2000.0  # Number of pixels in a row.
        factor = max_pixels / max(image.shape)
        if min(1.0, factor) < 1.0:
            image = cv2.resize(image, (0, 0), fx=factor, fy=factor)

        filename = "/home/hw/Desktop/mask_%i.png" % (index)
        cv2.imwrite(filename, image)
        print "File written to %s" % (filename)


def remove_water_in_slice_in_new_process(water_mask_bool, lat_indexes, lon_indexes,
                               orig_grid_res, min_lake_area_km2, output_queue, i):
    # Start the process.
    p = mp.Process(target=remove_water_in_slice,
                   args=(water_mask_bool[lat_indexes[0]:lat_indexes[1], lon_indexes[0]:lon_indexes[1]],
                         lats[lat_indexes[0]:lat_indexes[1]],
                         lons[lon_indexes[0]:lon_indexes[1]],
                         orig_grid_res,
                         min_lake_area_km2,
                         lat_indexes,
                         lon_indexes,
                         output_queue,
                         i))
    LOG.debug("Starting.")
    p.start()


def resample_mask(resulting_mask_with_removed_lakes, lats, lons, domain,
                  dlat, dlon, water_limit_percentage=0.1):
    """
    Resampling the mask to an new resolution, corresponding to dlat (delta
    lat) and dlon (delta lon).

    The input mask is the full resolution mask where the lakes have already
    been removed.

    The lats and the lons are the full resolution lats and lons corresponding
    to the values in the resulting_mask_with_removed_lakes.

    +--------+--------+--------+--------+--------+  0
    |        |        |        |        |        |
    +--------+--------+--------+--------+--------+  1
    |    -   |   -    |   -    |   -    |        |
    +--------+--------+--------+--------+--------+  2
    |    -   |   -    |   -    |   -    |        |
    +--------+--------+--------+--------+--------+  3
    |   -    |   x    |    -   |   -    |        |    lats
    +--------+--------+--------+--------+--------+  4
    |   -    |   -    |   -    |   -    |        |
    +--------+--------+--------+--------+--------+  5
    |   -    |   -    |   -    |   -    |        |
    +--------+--------+--------+--------+--------+  6
    |        |        |        |        |        |
    +--------+--------+--------+--------+--------+  7
    0        1        2        3        4        5
                         lons

    For every point that is checked (x), the surrounding points (-)
    will also count for that point. The number of points is determined
    by the resolution (dlat, dlon).

    If a certain percentage of the points is water
    (water_limit_percentage), the new point, with the new resolution,
    will also be water.
    """
    # Starting points.
    start_lat = max(domain.south, domain.north)
    stop_lat = min(domain.south, domain.north)

    # First create an "indexed list" and fill it with None.
    # This is used to insert the resampled data. This also means that
    # if there are no points somewhere, the data in that point is None.
    resampled_mask = []
    lat = start_lat
    while lat >= stop_lat:
        resampled_mask.append(None)
        lat -= dlat

    # Get the longitude limits.
    min_lon = min(domain.west, domain.east)
    max_lon = max(domain.west, domain.east)

    # A little setup.
    i = 0
    number_of_cpus = mp.cpu_count()
    output_queue = mp.Queue()

    # Start the northmost.
    lat = start_lat
    while lat >= stop_lat:
        print domain
        print lat, dlat
        print lats
        min_lat_index, max_lat_index = get_max_min_indexes(lats,
                                                           lat - dlat / 2.0,
                                                           lat + dlat / 2.0)

        print min_lat_index, max_lat_index
        # Start a new process.
        start_resample_in_new_process(resulting_mask_with_removed_lakes,
                                      lons,
                                      min_lon,
                                      max_lon,
                                      dlon,
                                      min_lat_index,
                                      max_lat_index,
                                      water_limit_percentage,
                                      output_queue,
                                      i)

        # Do not wait for resulting data before all the cpus are being used.
        # This means that there will be a rest of results in the output queue
        # after the while loop.
        if i >= number_of_cpus:
            # Append the row to the new mask.
            j, resulting_row = output_queue.get()
            resampled_mask[j] = resulting_row

        # Next row.
        lat -= dlat
        i += 1

    # Handle the rest in the output queue from the while loop.
    for i in range(number_of_cpus):
        j, resulting_row = output_queue.get()
        resampled_mask[j] = resulting_row

    # Convert the very resulting array to a boolean loop.
    # Here, all the missing data becomes False...
    resampled_array = np.array(resampled_mask, dtype=np.bool)
    return resampled_array


def start_resample_in_new_process(resulting_mask_with_removed_lakes,
                                  lons,
                                  start_lon,
                                  max_lon,
                                  dlon,
                                  min_lat_index,
                                  max_lat_index,
                                  water_limit_percentage,
                                  output_queue,
                                  i):
    # Start the process.
    p = mp.Process(target=resample_lon_row,
                   args=(resulting_mask_with_removed_lakes,
                         lons,
                         start_lon,
                         max_lon,
                         dlon,
                         min_lat_index,
                         max_lat_index,
                         water_limit_percentage,
                         output_queue,
                         i))
    LOG.debug("Starting fisk. %i"%(i))
    p.start()


def resample_lon_row(full_resolution_mask_with_removed_lakes, lons, start_lon, max_lon,
                     dlon, min_lat_index, max_lat_index, water_limit_percentage,
                     output_queue, i):
    """
    Resampling an area of grid points to a new grid point. 

    E.g. so that

    x--x--x--x--x--x--x--x--x--x--x--x--x--x--x--x--x--x--x--x--x--x
    |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
    x--x--x--x--x--x--x--x--x--x--x--x--x--x--x--x--x--x--x--x--x--x
    |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
    x--x--x--x--x--x--x--x--x--x--x--x--x--x--x--x--x--x--x--x--x--x

    becomes

    X-----X-----X-----X-----X-----X-----X-----X-----X-----X-----X


    if a certain percentage of the points the resampled mask is reprecenting
    (> water_limit_percentage) is water, the resampled point is considered water.
    Else it is considered land.
    """
    lon = start_lon
    resampled_lon_row = []

    #while lon <= max(domain.west, domain.east):
    while lon <= max_lon:
        min_lon_index, max_lon_index = get_max_min_indexes(lons,
                                                           lon - dlon / 2.0,
                                                           lon + dlon / 2.0)
        
        # Get the slice of data from the full resolution mask.
        full_resolution_mask_slice = full_resolution_mask_with_removed_lakes[min_lat_index:max_lat_index,
                                                                             min_lon_index:max_lon_index]

        # Make sure the indexes are correct.
        assert((max_lat_index - min_lat_index) *
               (max_lon_index - min_lon_index) ==
               full_resolution_mask_slice.size)

        # Calculate the water percentage from the surrounding points.
        # Number of nonzero (in this case True) in the slice divided by the
        # total number of items in the slice.
        water_percentage = np.count_nonzero(full_resolution_mask_slice) / float(full_resolution_mask_slice.size)

        # If there is more than a certain percentage of water in the area.
        resampled_lon_row.append(water_percentage > water_limit_percentage)

        # Next row
        lon += dlon
    output_queue.put([i, resampled_lon_row])



@decorators.timer
def remove_water_in_slice(water_mask_bool, lats, lons, orig_grid_res, min_lake_area_km2, lat_indexes, lon_indexes, output_queue, i):
    """
    Remove all the water of a certain size in the given water mask.
    """
    # Create a mask with only land (zero is False).
    resulting_mask = np.zeros(water_mask_bool.shape, dtype=np.bool)
    if np.all(water_mask_bool):
        # If everything is water, the mask with only is converted to 
        # be all water.
        resulting_mask = ~resulting_mask
    elif np.all(~water_mask_bool):
        # If nothing is water, then the new mask should just be land,
        # which it already is.
        pass
    else:
        # Create contours 
        for top_contour in Contours.get_top_contours(water_mask_bool):
            resulting_mask |= top_contour.remove_water_less_than(lats, lons, orig_grid_res, min_lake_area_km2)
    output_queue.put([i, lat_indexes, lon_indexes, resulting_mask])


if __name__ == "__main__":
    import docopt
    args = docopt.docopt(__doc__, version="Lake filter 0.1")

    # Set the log options.
    if args['--debug']:
        logging.basicConfig(filename=args['--log-filename'], level=logging.DEBUG)
    elif args['--verbose']:
        logging.basicConfig(filename=args['--log-filename'], level=logging.INFO)
    else:
        logging.basicConfig(filename=args['--log-filename'], level=logging.WARNING)

    # Output what is in the args variable.
    LOG.debug(args)

    # Make sure that the input sea mask exists.
    assert(os.path.isfile(args['<fine_land_sea_mask>']))

    # Convert the sea area to float
    args['<min_lake_area_km2>'] = float(args['<min_lake_area_km2>'])
    # and make sure that it is a positive number!
    assert(args['<min_lake_area_km2>'] > 0)

    # Set the to the delta lat and delta lon.
    if args['--resolution'] is not None:
        args['--dlat'] = args['--resolution']
        args['--dlon'] = args['--resolution']

    # Convert the resolutions to float.
    if args['--dlat'] is not None:
        args['--dlat'] = float(args['--dlat'])
        args['--dlon'] = float(args['--dlon'])

    # Determine the domain.
    if args['--domain'] is not None:
        assert(args['--domain'] in domain.DOMAINS)
        domain = domain.DOMAINS[args['--domain'].upper()]
    elif args['<north>'] is not None:
        # Doc opt handles that the rest must be set as well...
        domain = domain.Domain("Custom",
                               args['<west>'],
                               args['<east>'],
                               args['<south>'],
                               args['<north>']
                               )
    else:
        # Use global domain as default.
        domain = domain.DOMAINS["GLB"]

    LOG.debug("Read input file, %s." % (args['<fine_land_sea_mask>']))
    rootgrp = netCDF4.Dataset(args['<fine_land_sea_mask>'], 'r')
    try:
        # This is the grid resolution of the grids in the input land sea
        # mask netcdf file.
        orig_grid_res = float(rootgrp.grid_resolution.split("degree")[0])

        LOG.debug("All the variables in the file %s: %s" % (args['<fine_land_sea_mask>'],
                                                            rootgrp.variables))

        # Land/sea mask with distance from land.
        # I.e. everything > 0, is water.
        LOG.debug("Getting the land/sea mask.")
        distance_to_land_mask = rootgrp.variables['dst'][:]
        lats = rootgrp.variables['lat'][:]
        lons = rootgrp.variables['lon'][:]

        lat_min_index, \
            lat_max_index, \
            lon_min_index, \
            lon_max_index = get_lat_lon_indexes(lats, # All the lats values.
                                                lons, # All the lons values.
                                                domain.south, # From the input.
                                                domain.north, # From the input.
                                                domain.west,  # From the input.
                                                domain.east,  # From the input.
                                                args['--dlat'], # Resolution of the output.
                                                args['--dlon']) # Resolution of the output.
    
        # WHY??
        """
        if lat_min_index == 0:
            domain.south = min(lats) + args['--dlat']
        if lat_max_index == len(lats) - 1:
            domain.north = max(lats) - args['--dlat']

        if lon_min_index == 0:
            domain.west = min(lons) + args['--dlon']
        if lon_max_index == len(lons) - 1:
            domain.east = max(lons) - args['--dlon']
            """

        resulting_mask_with_removed_lakes = np.zeros((lat_max_index - lat_min_index,
                                                      lon_max_index - lon_min_index),
                                                     dtype=np.bool)
        # global_mask_with_removed_lakes = int_water_mask_2_bool_water_mask(global_mask_with_removed_lakes)

        number_of_cpus = mp.cpu_count()

        output_queue = mp.Queue()
        slice_counter = 0

        LOG.debug("Converting water mask to boolean.")
        water_mask_bool = int_water_mask_2_bool_water_mask(distance_to_land_mask[lat_min_index:lat_max_index, lon_min_index:lon_max_index])
        LOG.debug("Getting lats.")
        lats = lats[lat_min_index:lat_max_index]
        LOG.debug("Getting lons.")
        lons = lons[lon_min_index:lon_max_index]

        # Start the calculations.
        for lat_slice_indexes, lon_slice_indexes in get_slice_indexes(0, len(lats), 0, len(lons)):
            slice_counter += 1
            # Start a new process that removes water.
            remove_water_in_slice_in_new_process(water_mask_bool, lat_slice_indexes, lon_slice_indexes, orig_grid_res, args['<min_lake_area_km2>'], output_queue, slice_counter)

            # Wait for result.
            # But wait until it has started as many processes as there are CPUs.
            # This means that when the outer for loop has finished, there will be
            # some results left in the queue. See [*] below.
            if slice_counter > number_of_cpus:
                LOG.debug("Waiting for output queue.")
                i, lat_slice_indexes, lon_slice_indexes, mask = output_queue.get()
                LOG.debug("Got %i" % (i))
                # Merge the masks.
                resulting_mask_with_removed_lakes[lat_slice_indexes[0]:lat_slice_indexes[1],
                                                  lon_slice_indexes[0]:lon_slice_indexes[1]] |= mask

        # [*] Take care of the results left in the queue.
        for j in range(number_of_cpus):
            LOG.debug("Waiting for output queue.")
            i, lat_slice_indexes, lon_slice_indexes, mask = output_queue.get()
            LOG.debug("Got %i" % (i))
            # Merge the masks.
            resulting_mask_with_removed_lakes[lat_slice_indexes[0]:lat_slice_indexes[1],
                                              lon_slice_indexes[0]:lon_slice_indexes[1]] |= mask

    finally:
        rootgrp.close()

    # Create the new mask, with new indexes.
    new_mask = resample_mask(resulting_mask_with_removed_lakes,
                             lats, lons, domain, args['--dlat'], args['--dlon'])

    LOG.debug("Showing result.")
    show_masks([new_mask, ])
    # show_masks([resulting_mask_with_removed_lakes, water_mask_bool])

    LOG.debug("DONE")
