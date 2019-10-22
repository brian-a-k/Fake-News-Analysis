from typing import *
import pandas as pd
import numpy as np


def clean_columns(raw_cols: List[str]) -> List[str]:
    return [name.lower().replace(' ', '_') for name in raw_cols]

