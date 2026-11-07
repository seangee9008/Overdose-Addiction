-- TEDS-A admissions aggregated to state x year (area level). Opioid groupings use the PRIMARY substance
-- (SUB1): 5 heroin, 6 non-prescription methadone, 7 other opiates and synthetics (includes fentanyl,
-- prescription opioids and buprenorphine; the public file has no fentanyl-specific code). any_opioid =
-- heroin / non-Rx methadone / other opiates reported as primary, secondary or tertiary substance.
select
    t.state_fips,
    t.year,
    count(*)                                                            as total_admissions,
    count(*) filter (where t.sub1 in (5, 6, 7))                         as primary_opioid,
    count(*) filter (where t.sub1 = 5)                                  as primary_heroin,
    count(*) filter (where t.sub1 = 6)                                  as primary_methadone_nonrx,
    count(*) filter (where t.sub1 = 7)                                  as primary_other_opiates,
    count(*) filter (where t.herflg = 1 or t.methflg = 1 or t.opsynflg = 1) as any_opioid,
    count(*) filter (where t.sub1 is null)                              as primary_substance_missing,
    t.state_fips || '|' || cast(t.year as varchar)                      as row_key
from {{ ref('stg_teds_a') }} t
group by t.state_fips, t.year
