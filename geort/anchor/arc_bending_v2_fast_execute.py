"""Run the approved v2 pipeline with exact CPU-accelerated medoids."""

from geort.anchor import arc_bending_v2_robust_execute as _runner
from geort.anchor.arc_bending_v2_fast import select_robust_arc_medoids_fast


_runner.select_robust_arc_medoids = select_robust_arc_medoids_fast
main = _runner.main


if __name__ == "__main__":
    main()
