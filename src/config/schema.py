
#Prevents silent training bugs

#Extremely impressive in reviews/interviews

#Ray-native (PyArrow schema)


from typing import Dict
import pyarrow as pa

FEATURE_COLUMNS = ["f1", "f2", "f3", "f4"]
TARGET_COLUMN = "target"

def feature_schema() -> pa.schema:
    return pa.schema(
        [
            ("f1",pa.float32()),
            ("f2",pa.float32()),
            ("f3",pa.float32()),
            ("f4",pa.float32()),
            ("target",pa.int64()),
        ]
    )


def validate_schema(dataset) -> None:
    actual = dataset.schema()
    expected = feature_schema()

    if actual != feature_schema:
        raise ValueError(
            f"Schema mismatch.\nExpected:\n{expected}\nActual:\n{actual}"
        )