import ray
import ray.data as rd
import pandas as pd

# Initialize Ray
ray.init(ignore_reinit_error=True)

# Create test data
data = {
    "id": [1, 2, 3],
    "feature_1_norm": [1.0, 2.0, 3.0],
    "feature_2_norm": [4.0, 5.0, 6.0],
    "feature_3": [7.0, 8.0, 9.0],
    "feature_4": [10.0, 11.0, 12.0],
    "target": [0, 1, 0]
}

df = pd.DataFrame(data)
dataset = rd.from_pandas(df)

# Check what type of schema we get
schema = dataset.schema()
print(f"Schema type: {type(schema)}")
print(f"Schema: {schema}")

# Check available methods
print(f"\nAvailable methods and attributes:")
for attr in dir(schema):
    if not attr.startswith('_'):
        print(f"  {attr}")

# Try to convert to PyArrow
if hasattr(schema, 'to_arrow_schema'):
    print(f"\nHas to_arrow_schema method")
    arrow_schema = schema.to_arrow_schema()
    print(f"Arrow schema type: {type(arrow_schema)}")
    print(f"Arrow schema field names: {arrow_schema.names}")
    
if hasattr(schema, 'arrow_schema'):
    print(f"\nHas arrow_schema attribute")
    arrow_schema = schema.arrow_schema
    print(f"Arrow schema type: {type(arrow_schema)}")

ray.shutdown()
