# Getting Started

This getting started guide demonstrates the core functionality of the project-lighthouse-anonymize package using the Adults dataset from [the UC Irvine Machine Learning Repository](https://archive.ics.uci.edu/) using the python package [ucimlrepo](https://github.com/uci-ml-repo/ucimlrepo).

For background on the privacy models demonstrated here, see the foundational Project Lighthouse paper: [Measuring discrepancies in Airbnb guest acceptance rates using anonymized demographic data](https://news.airbnb.com/wp-content/uploads/sites/4/2020/06/Project-Lighthouse-Airbnb-2020-06-12.pdf).

## Installation

Install the required dependencies to get started:

```bash
pip install -U --quiet ucimlrepo project-lighthouse-anonymize
```

> **Note:** This package supports Python 3.9-3.12. Python 3.13+ is not yet supported due to the numpy<2.0.0 constraint (NumPy 2.0+ is required for Python 3.13 wheels).

## Setup logging

Set up logging to track the anonymization process.

```python
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
```

## Load Adult Census Income dataset

Load and explore the Adult Census Income dataset from UCI ML Repository.

```python
import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo

adult = fetch_ucirepo(id=2)
X = adult.data.features
y = adult.data.targets
adult_df = pd.concat([X, y], axis=1)

adult_df.describe()
```

```
               age        fnlwgt  education-num  capital-gain  capital-loss  hours-per-week
count  48842.000000  4.884200e+04   48842.000000  48842.000000  48842.000000    48842.000000
mean      38.643585  1.896641e+05      10.078089   1079.067626     87.502314       40.422382
std       13.710510  1.056040e+05       2.570973   7452.019058    403.004552       12.391444
min       17.000000  1.228500e+04       1.000000      0.000000      0.000000        1.000000
25%       28.000000  1.175505e+05       9.000000      0.000000      0.000000       40.000000
50%       37.000000  1.781445e+05      10.000000      0.000000      0.000000       40.000000
75%       48.000000  2.376420e+05      12.000000      0.000000      0.000000       45.000000
max       90.000000  1.490400e+06      16.000000  99999.000000   4356.000000       99.000000
```

## Pre-process dataset

Clean the data by standardizing income labels, handling missing values, and adding row identifiers.

```python
adult_df["income"].replace("<=50K.", "<=50K", inplace=True)
adult_df["income"].replace(">50K.", ">50K", inplace=True)

for col in adult_df.columns:
    adult_df.loc[adult_df[col] == "?", col] = np.nan

adult_df = adult_df.reset_index(drop=True)
adult_df["row_id"] = list(range(len(adult_df)))

adult_df.describe()
```

```
               age        fnlwgt  education-num  capital-gain  capital-loss  hours-per-week        row_id
count  48842.000000  4.884200e+04   48842.000000  48842.000000  48842.000000    48842.000000  48842.000000
mean      38.643585  1.896641e+05      10.078089   1079.067626     87.502314       40.422382  24420.500000
std       13.710510  1.056040e+05       2.570973   7452.019058    403.004552       12.391444  14099.615261
min       17.000000  1.228500e+04       1.000000      0.000000      0.000000        1.000000      0.000000
25%       28.000000  1.175505e+05       9.000000      0.000000      0.000000       40.000000  12210.250000
50%       37.000000  1.781445e+05      10.000000      0.000000      0.000000       40.000000  24420.500000
75%       48.000000  2.376420e+05      12.000000      0.000000      0.000000       45.000000  36630.750000
max       90.000000  1.490400e+06      16.000000  99999.000000   4356.000000       48841.000000
```

```python
filter_row_id = lambda row_id: row_id % 7000 == 31
adult_df[adult_df["row_id"].apply(filter_row_id)].sort_values('row_id')
```

```
       age         workclass    fnlwgt     education  education-num      marital-status         occupation   relationship   race     sex  capital-gain  capital-loss  hours-per-week native-country income  row_id
31     20.0           Private  266015.0  Some-college           10.0       Never-married              Sales      Own-child  Black    Male           0.0           0.0            44.0  United-States  <=50K      31
7031   44.0           Private  104440.0     Bachelors           13.0  Married-civ-spouse    Exec-managerial        Husband  White    Male           0.0           0.0            40.0  United-States   >50K    7031
14031  52.0  Self-emp-not-inc  190333.0       HS-grad            9.0            Divorced    Farming-fishing  Not-in-family  White    Male           0.0           0.0            25.0  United-States  <=50K   14031
21031  22.0           Private  194031.0  Some-college           10.0       Never-married      Other-service      Own-child  White  Female           0.0           0.0            15.0  United-States  <=50K   21031
28031  33.0           Private   85355.0       HS-grad            9.0           Separated  Machine-op-inspct  Not-in-family  White    Male           0.0           0.0            30.0  United-States  <=50K   28031
35031  49.0           Private   90579.0       HS-grad            9.0  Married-civ-spouse       Craft-repair        Husband  Black    Male        5013.0           0.0            50.0  United-States  <=50K   35031
42031  43.0           Private   77373.0     Bachelors           13.0  Married-civ-spouse    Exec-managerial        Husband  White    Male           0.0           0.0            47.0  United-States   >50K   42031

[7 rows x 16 columns]
```

## Enforce first technical privacy model, k-anonymity

Apply k-anonymization using the Core Mondrian algorithm. The `k_anonymize` wrapper function provides easy access to the anonymization functionality with comprehensive data quality and disclosure risk metrics.

Note that we limit the columns examined because k-anonymity cannot be achieved while meeting our minimums as defined in `default_dq_metric_to_minimum_dq`, due to the high dimensionality of the dataset.

```python
from pprint import pprint
from project_lighthouse_anonymize import k_anonymize, check_dq_meets_minimum_thresholds

qids = ["age", "hours-per-week", "workclass", "native-country", "income"]
k = 5

input_df = adult_df[qids + ["row_id", "race"]].copy()

anon_df, dq_metrics, disclosure_metrics = k_anonymize(logger, input_df, qids, k, {}, "row_id")
```

```
2025-09-09 08:21:19,987 - INFO - Anonymizing using k-anonymity with k = 5
2025-09-09 08:21:20,034 - INFO - numerical_rilm_domains = {'age': [< [10.0%, 90.0%] => [22.0, 58.0] >,
        < [1.0%, 99.0%] => [17.0, 74.0] >,
        < [0.1%, 99.9%] => [17.0, 90.0] >,
        < [0.01%, 99.99%] => [17.0, 90.0] >,
        < [0.001%, 99.999%] => [17.0, 90.0] >,
        < [0.0%, 100.0%] => [17.0, 90.0] >],
'hours-per-week': [< [10.0%, 90.0%] => [24.0, 55.0] >,
                   < [1.0%, 99.0%] => [8.0, 80.0] >,
                   < [0.1%, 99.9%] => [2.0, 99.0] >,
                   < [0.01%, 99.99%] => [1.0, 99.0] >,
                   < [0.001%, 99.999%] => [1.0, 99.0] >,
                   < [0.0%, 100.0%] => [1.0, 99.0] >]}
```

```python
pprint(dq_metrics)
```

```
{'n_generalized': 17698,
 'n_non_suppressed': 48493,
 'n_suppressed': 349,
 'nmi_sampled_scaled_v1__age': 0.9321025201597688,
 'nmi_sampled_scaled_v1__hours-per-week': 0.8997919098117235,
 'nmi_sampled_scaled_v1__minimum': 0.8997919098117235,
 'pct_generalized': 0.3623520740346423,
 'pct_non_suppressed': 0.992854510462307,
 'pct_suppressed': 0.007145489537692969,
 'pearsons__age': 0.9928571671034669,
 'pearsons__hours-per-week': 0.9815827849498265,
 'pearsons__minimum': 0.9815827849498265,
 'rilm__income': 0.9795021961932651,
 'rilm__native-country': 0.9871967089245236,
 'rilm__workclass': 0.9996719160104987,
 'rilm_categorical__minimum': 0.9795021961932651}
```

```python
pprint(disclosure_metrics)
```

```
{'actual_k': 5}
```

```python
minimum_dq_met, minimum_dq_met_reasons = check_dq_meets_minimum_thresholds(dq_metrics)
assert minimum_dq_met, str(minimum_dq_met_reasons)

anon_df.describe()
```

```
               age  hours-per-week        row_id
count  48493.000000    48493.000000  48493.000000
mean      38.664178       40.462747  24417.559442
std       13.609099       12.141260  14100.085496
min       17.000000        2.000000      0.000000
25%       28.000000       39.000000  12207.000000
50%       37.000000       40.000000  24415.000000
75%       48.000000       45.000000  36629.000000
max       89.923077       99.000000  48841.000000
```

```python
anon_df[anon_df["row_id"].apply(filter_row_id)].sort_values('row_id')
```

```
            age  hours-per-week         workclass native-country income  row_id   race
31     20.750000            44.0           Private  United-States  <=50K      31  Black
7031   44.000000            40.0           Private  United-States   >50K    7031  White
14031  53.833333            24.5  Self-emp-not-inc  United-States  <=50K   14031  White
21031  22.000000            15.0           Private  United-States  <=50K   21031  White
28031  33.000000            30.0           Private  United-States  <=50K   28031  White
35031  49.000000            50.0           Private  United-States  <=50K   35031  Black
42031  41.800000            46.4           Private  United-States   >50K   42031  White
```

Note that the `check_dq_meets_minimum_thresholds` function passes successfully with this reduced set of quasi-identifiers, demonstrating that data quality requirements can be met when the dimensionality is appropriately managed.

**Observe the microaggregation effects**: Compare the original sample values with the k-anonymized results. Notice how numerical quasi-identifiers are averaged within equivalence classes:
- Row 31: age 20.0 → 20.75 (slight increase due to grouping)
- Row 14031: age 52.0 → 53.83, hours-per-week 25.0 → 24.50
- Row 42031: age 43.0 → 41.80, hours-per-week 47.0 → 46.40

These changes preserve privacy by making individual records indistinguishable within their k-anonymous groups while maintaining data utility for analysis.

## Enforce second technical privacy model, p-sensitive k-anonymity

Apply p-sensitive anonymization to ensure diversity in sensitive attributes. The `p_sensitize` wrapper function converts k-anonymous datasets to p-sensitive k-anonymous datasets through controlled perturbations.

Note that we apply p-sensitive anonymization as a separate step after k-anonymization because in production environments, the sensitive attribute (race) is not available until after the first privacy model is enforced—see the foundational paper for more details on this system design.

```python
from project_lighthouse_anonymize import p_sensitize

p, k = 2, 5
sens_attr_value_to_prob = {"Amer-Indian-Eskimo": 0.20, "Asian-Pac-Islander": 0.20, "Black": 0.20, "Other": 0.20, "White": 0.20}

sensitized_df, dq_metrics, disclosure_metrics = p_sensitize(logger, anon_df, qids, "race", p, k, sens_attr_value_to_prob, seed=123)
```

```python
pprint(dq_metrics)
```

```
{'num_rows_perturbated': 1482, 'pct_perturbated': 3.0561111913059618}
```

```python
pprint(disclosure_metrics)
```

```
{'actual_k': 5, 'actual_p': 2}
```

```python
sensitized_df.describe()
```

```
               age  hours-per-week        row_id
count  48493.000000    48493.000000  48493.000000
mean      38.664178       40.462747  24417.559442
std       13.609099       12.141260  14100.085496
min       17.000000        2.000000      0.000000
25%       28.000000       39.000000  12207.000000
50%       37.000000       40.000000  24415.000000
75%       48.000000       45.000000  36629.000000
max       89.923077       99.000000  48841.000000
```

```python
sensitized_df[sensitized_df["row_id"].apply(filter_row_id)].sort_values('row_id')
```

```
            age  hours-per-week         workclass native-country income  row_id                race
31     20.750000            44.0           Private  United-States  <=50K      31               Black
7031   44.000000            40.0           Private  United-States   >50K    7031               White
14031  53.833333            24.5  Self-emp-not-inc  United-States  <=50K   14031               White
21031  22.000000            15.0           Private  United-States  <=50K   21031               White
28031  33.000000            30.0           Private  United-States  <=50K   28031               White
35031  49.000000            50.0           Private  United-States  <=50K   35031               Black
42031  41.800000            46.4           Private  United-States   >50K   42031  Amer-Indian-Eskimo
```

**Observe the p-sensitive anonymization effects**: Compare the k-anonymized sample with the p-sensitized results. The key change to notice:
- Row 42031: race "White" → "Amer-Indian-Eskimo"

This demonstrates how p-sensitive k-anonymity works: the algorithm identifies equivalence classes that lack sufficient diversity in the sensitive attribute (race) and strategically modifies some values to ensure each group contains at least p=2 distinct sensitive values. All other quasi-identifier values (age, hours-per-week, workclass, etc.) remain unchanged from the k-anonymization step, preserving the privacy guarantees while adding sensitive attribute diversity.
