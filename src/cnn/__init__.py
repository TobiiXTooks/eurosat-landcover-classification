"""From-scratch CNN pipeline for EuroSAT classification.

The package trains and evaluates scratch CNN variants on two tasks over the
EuroSAT RGB dataset:

* ``multiclass`` -- the 10-class land-use classification.
* ``binary`` -- vegetation vs non-vegetation classification.

Both tasks reuse a single stratified split (seed 42) so model comparisons rest
on identical data partitions.
"""

__version__ = "0.1.0"
