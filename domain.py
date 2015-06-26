
# coding: utf-8
import logging
LOG = logging.getLogger(__name__)


class Domain:
    def __init__(self, name, west, east, south, north):
        self.name = name
        self.west = float(west)
        self.east = float(east)
        self.south = float(south)
        self.north = float(north)

    def __str__(self):
        return "%s: %0.3f %0.3f %0.3f %0.3f" % (
            self.name,
            self.west,
            self.east,
            self.south,
            self.north,
            )

DOMAINS = {
    "ARC": Domain("Arctic", -179.995, 179.955, 58.000, 88.000),
    "GBL": Domain("Global", -179.975, 179.975, -89.000,  89.000),
    "NSB": Domain("North sea baltic", -12.000,  32.000,  46.000, 68.000)
    }


