# Parameterised object references

`reference_objects.csv` converts the ruler dimensions encoded in the image
filenames into the RealSense contract: support-plane footprint length,
footprint width, and perpendicular height, all in millimetres.

Only plastic bags, paper bags, and cardboard boxes/cartons are accepted. The
backpack, laptop sleeve, and fabric laundry hamper are negative controls and
must produce no live track, dimensions, volume, or deposited history.

For rigid cuboids, `reference_volume_liters` is the external bounding volume
calculated from the supplied ruler dimensions. For deformable bags it is left
blank: bounding dimensions are useful dimension truth but do not establish
the volume occupied by irregular contents. A bag volume calibration requires
an independently known filled volume.

Use one object at a time, keep it still on the same support surface, and record
at least 10 RealSense samples with `scripts/validate_known_volume.py`. Do not
use Logitech dimensions as physical ground truth.
