"""The study window, declared once.

Five modules used to carry their own copy of ``list(range(2011, 2026))``. They
agreed with each other and disagreed with the database, which holds the
2011--2026 season window the manuscript reports. The off-by-one at the top of
the range was invisible because ``INSERT OR REPLACE`` never deletes a row: the
2026 cross-section written by an earlier run outlived the range that produced
it, so the manuscript counted 16 cross-sections, the prose gate froze the
literal ``16``, and the script that is supposed to regenerate both could only
ever write 15. A rebuild would have moved the national median exponent from
2.03 back to 2.00 without a single line of text changing.

The window lives here, with no third-party imports, so the fitting script, the
dashboard helpers and the table generator can share one definition without
dragging pandas or geopandas into the audit.
"""

#: First season year. 2010 is present in the weekly series but its season is not
#: decomposed, so it enters no cross-section.
FIRST_SEASON_YEAR = 2011

#: Last season year. Bounded by the data cut-off, so the last season is shorter
#: than the others; the Methods section says so rather than leaving the reader
#: to infer it from a row count.
LAST_SEASON_YEAR = 2026

#: The weekly series starts a year before the first season: a season has to be
#: observed beginning, and the decomposition needs the weeks before it.
FIRST_WEEKLY_YEAR = 2010

#: Season years that carry a cross-section, in order.
SEASON_YEARS = list(range(FIRST_SEASON_YEAR, LAST_SEASON_YEAR + 1))

#: Years of weekly data to fetch and aggregate.
WEEKLY_YEARS = list(range(FIRST_WEEKLY_YEAR, LAST_SEASON_YEAR + 1))

#: ``year = 0`` is the pooled all-years fit, not a calendar year. It is excluded
#: from every year count and must survive any pruning of out-of-window rows.
POOLED_YEAR = 0
