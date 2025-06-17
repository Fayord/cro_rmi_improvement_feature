import pydantic
from typing import Type, Any, Dict, List, Union, Literal
import json


# 1. Function to create a Pydantic model dynamically
def create_dynamic_pydantic_model(
    model_name: str, fields: Dict[str, Union[Type, tuple[Type, Any]]]
) -> Type[pydantic.BaseModel]:
    """
    Creates a Pydantic BaseModel class dynamically, allowing for field descriptions
    and other Pydantic.Field metadata.

    Args:
        model_name (str): The desired name for the dynamic Pydantic model.
        fields (Dict[str, Union[Type, tuple[Type, Any]]]):
            A dictionary where keys are field names (str) and values are either:
            - Their corresponding Python type (e.g., str, int)
            - A tuple (Python type, pydantic.Field object) to include metadata like description.

    Returns:
        Type[pydantic.BaseModel]: A dynamically created Pydantic model class.
    """
    processed_fields = {}
    for field_name, field_definition in fields.items():
        if isinstance(field_definition, tuple):
            # If it's a tuple, it should be (type, Field object)
            field_type, field_info = field_definition
            if not isinstance(field_info, pydantic.fields.FieldInfo):
                # Ensure the second element of the tuple is a Pydantic Field
                raise TypeError(
                    f"For field '{field_name}', the second element of the tuple "
                    "must be a pydantic.Field object, not {type(field_info)}"
                )
            processed_fields[field_name] = (field_type, field_info)
        else:
            # Otherwise, it's just a type
            processed_fields[field_name] = field_definition

    # Use pydantic.create_model to generate the model class
    # This function takes the model name, base class, and field definitions.
    # The processed_fields are unpacked.
    return pydantic.create_model(model_name, **processed_fields)


# --- Example Usage ---

print("--- Example 1: Basic Dynamic Model Creation with Descriptions ---")
# Define the fields for our first dynamic model, now including descriptions
fields_data_1 = {
    "name": (str, pydantic.Field(description="The full name of the person")),
    "age": (
        int,
        pydantic.Field(ge=0, description="The age of the person, must be non-negative"),
    ),
    "email": (
        str,
        pydantic.Field(
            pattern=r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$",
            description="The person's email address",
        ),
    ),
}

# Create the dynamic model
PersonModel = create_dynamic_pydantic_model("PersonModel", fields_data_1)

# Instantiate the model with data that conforms to its schema
try:
    person_instance = PersonModel(
        name="Alice Smith", age=30, email="alice.smith@example.com"
    )
    print(f"PersonModel instance 1: {person_instance.model_dump_json(indent=2)}")
except pydantic.ValidationError as e:
    print(f"Validation Error: {e}")

# Try to instantiate with incorrect data to see validation in action
print(
    "\nAttempting to create PersonModel with invalid data (negative age and invalid email):"
)
try:
    invalid_person_instance = PersonModel(
        name="Bob Johnson", age=-5, email="bob@invalid"
    )
    print(
        f"Invalid PersonModel instance: {invalid_person_instance.model_dump_json(indent=2)}"
    )
except pydantic.ValidationError as e:
    print(f"Validation Error (expected): {e}")


print("\n--- Example 2: Dynamic Models in a Loop with Varying Descriptions ---")
# List of different schema definitions, some with descriptions
schema_definitions: List[Dict[str, Any]] = [
    {
        "model_name": "Product",
        "fields": {
            "product_id": (
                int,
                pydantic.Field(description="Unique identifier for the product"),
            ),
            "name": (str, pydantic.Field(description="Name of the product")),
            "price": (
                float,
                pydantic.Field(
                    gt=0, description="Price of the product, must be greater than zero"
                ),
            ),
        },
    },
    {
        "model_name": "Order",
        "fields": {
            "order_id": (
                str,
                pydantic.Field(description="Unique identifier for the order"),
            ),
            "customer_id": int,  # This field does not have a description
            "items": (
                List[str],
                pydantic.Field(
                    description="List of product names included in the order"
                ),
            ),
            "total_amount": (
                float,
                pydantic.Field(ge=0, description="Total cost of the order"),
            ),
        },
    },
    {
        "model_name": "SensorReading",
        "fields": {
            "sensor_type": (
                str,
                pydantic.Field(
                    description="Type of sensor (e.g., temperature, humidity)"
                ),
            ),
            "value": float,
            "timestamp": (
                str,
                pydantic.Field(description="Time of the reading in ISO format"),
            ),
        },
    },
]

dynamic_models = {}

# Loop through the schema definitions to create and use models dynamically
for i, schema_def in enumerate(schema_definitions):
    model_name = schema_def["model_name"]
    fields = schema_def["fields"]

    print(
        f"\nLoop {i+1}: Creating model '{model_name}' with fields and descriptions..."
    )

    # Create the dynamic model for the current iteration
    try:
        CurrentDynamicModel = create_dynamic_pydantic_model(model_name, fields)
        dynamic_models[model_name] = CurrentDynamicModel
        # You can inspect the generated schema including descriptions
        print(f"  {model_name} JSON Schema (excerpt):")
        # Print only relevant parts of the schema for brevity
        for prop_name, prop_details in (
            CurrentDynamicModel.model_json_schema().get("properties", {}).items()
        ):
            print(
                f"    - {prop_name}: {prop_details.get('type')}"
                + (
                    f" (Description: {prop_details.get('description')})"
                    if "description" in prop_details
                    else ""
                )
            )

        # Example: Use the dynamically created model
        if model_name == "Product":
            product_data = {"product_id": 101, "name": "Laptop Pro", "price": 1800.00}
            try:
                product = CurrentDynamicModel(**product_data)
                print(f"  {model_name} instance: {product.model_dump_json(indent=2)}")
            except pydantic.ValidationError as e:
                print(f"  Validation Error for Product: {e}")
        elif model_name == "Order":
            order_data = {
                "order_id": "ORD002",
                "customer_id": 502,
                "items": ["Laptop Pro", "Mouse"],
                "total_amount": 1850.75,
            }
            try:
                order = CurrentDynamicModel(**order_data)
                print(f"  {model_name} instance: {order.model_dump_json(indent=2)}")
            except pydantic.ValidationError as e:
                print(f"  Validation Error for Order: {e}")
        elif model_name == "SensorReading":
            sensor_data = {
                "sensor_type": "humidity",
                "value": 65.2,
                "timestamp": "2025-06-13T10:30:00Z",
            }
            try:
                reading = CurrentDynamicModel(**sensor_data)
                print(f"  {model_name} instance: {reading.model_dump_json(indent=2)}")
            except pydantic.ValidationError as e:
                print(f"  Validation Error for SensorReading: {e}")
    except TypeError as e:
        print(f"  Error creating model '{model_name}': {e}")


print("\nAll dynamically created models (with descriptions):", dynamic_models)

# You can access a model and its schema later to see descriptions
# print("\nFull JSON Schema for Product model:")
# print(dynamic_models["Product"].model_json_schema(indent=2))


def create_edge_relationship_model(
    risk_a: str, risk_b: str
) -> Type[pydantic.BaseModel]:
    """
    Creates an EdgeRelationship model with specific risk names.

    Args:
        risk_a (str): Name of the first risk
        risk_b (str): Name of the second risk

    Returns:
        Type[pydantic.BaseModel]: A Pydantic model class for the edge relationship
    """
    schema_definition_dict = {
        "model_name": "EdgeRelationship",
        "fields": {
            "relationship": (
                Literal[
                    f"{risk_a}_caused_by_{risk_b}",
                    f"{risk_b}_caused_by_{risk_a}",
                    "no_relationship",
                    "be_a_cause_to_each_other",
                ],
                pydantic.Field(
                    description=f"Relationship between two risks. {risk_a}_caused_by_{risk_b} means {risk_b} is the cause and {risk_a} is the effect. {risk_b}_caused_by_{risk_a} means {risk_a} is the cause and {risk_b} is the effect. no_relationship means there is no relationship between two risks. be_a_cause_to_each_other means two risks are both cause and effect to each other.",
                ),
            ),
            "reason": (
                str,
                pydantic.Field(
                    description="Brief reason for the relationship classification in 1-3 sentences"
                ),
            ),
        },
    }

    return create_dynamic_pydantic_model(
        schema_definition_dict["model_name"],
        schema_definition_dict["fields"],
    )


# Example usage with different risk pairs
risk_pairs = [
    ("RISK_A", "RISK_B"),
    ("MARKET_RISK", "CREDIT_RISK"),
    ("OPERATIONAL_RISK", "COMPLIANCE_RISK"),
]

for risk_a, risk_b in risk_pairs:
    print("\n" + "=" * 50)
    print(f"Testing model for {risk_a} and {risk_b}:")
    model = create_edge_relationship_model(risk_a, risk_b)

    # Test 1: Valid relationship
    print("\nTest 1: Valid relationship")
    test_data = {
        "relationship": f"{risk_a}_caused_by_{risk_b}",
        "reason": f"{risk_b} is the cause and {risk_a} is the effect",
    }

    try:
        instance = model(**test_data)
        print("✅ Success: Valid instance created")
        print(f"Data: {instance.model_dump_json(indent=2)}")
    except pydantic.ValidationError as e:
        print("❌ Error: This should not happen with valid data")
        print(f"Error: {e}")

    # Test 2: Invalid relationship
    print("\nTest 2: Invalid relationship")
    invalid_data = {
        "relationship": "invalid_relationship",
        "reason": "This should fail validation",
    }

    try:
        instance = model(**invalid_data)
        print("❌ Error: This should not happen with invalid data")
        print(f"Data: {instance.model_dump_json(indent=2)}")
    except pydantic.ValidationError as e:
        print("✅ Success: Validation error caught as expected")
        print(f"Error: {e}")


def create_single_label_model() -> Type[pydantic.BaseModel]:
    """
    Creates a model with a single_label field that can only be choice_a, choice_b, or choice_c.
    Each choice has its own specific description.

    Returns:
        Type[pydantic.BaseModel]: A Pydantic model class for single label selection
    """
    schema_definition_dict = {
        "model_name": "SingleLabelModel",
        "fields": {
            "single_label": (
                Literal["choice_a", "choice_b", "choice_c"],
                pydantic.Field(
                    description=f"Select one of the predefined choices{json.dumps({
                        "choices": {
                            "choice_a": "This is the first choice, representing option A",
                            "choice_b": "This is the second choice, representing option B",
                            "choice_c": "This is the third choice, representing option C",
                        }
                    })}",
                    # json_schema_extra=,
                ),
            ),
        },
    }

    return create_dynamic_pydantic_model(
        schema_definition_dict["model_name"],
        schema_definition_dict["fields"],
    )


# Test the single label model
print("\n" + "=" * 50)
print("Testing Single Label Model:")

single_label_model = create_single_label_model()

# Print the model's JSON schema to see the descriptions
print("\nModel Schema:")
print(json.dumps(single_label_model.model_json_schema(), indent=2))

# Test 1: Valid choice
print("\nTest 1: Valid choice")
valid_data = {"single_label": "choice_a"}

try:
    instance = single_label_model(**valid_data)
    print("✅ Success: Valid instance created")
    print(f"Data: {instance.model_dump_json(indent=2)}")
except pydantic.ValidationError as e:
    print("❌ Error: This should not happen with valid data")
    print(f"Error: {e}")

# Test 2: Invalid choice
print("\nTest 2: Invalid choice")
invalid_data = {"single_label": "invalid_choice"}

try:
    instance = single_label_model(**invalid_data)
    print("❌ Error: This should not happen with invalid data")
    print(f"Data: {instance.model_dump_json(indent=2)}")
except pydantic.ValidationError as e:
    print("✅ Success: Validation error caught as expected")
    print(f"Error: {e}")

# Test 3: Try all valid choices
print("\nTest 3: All valid choices")
valid_choices = ["choice_a", "choice_b", "choice_c"]

for choice in valid_choices:
    print(f"\nTrying choice: {choice}")
    try:
        instance = single_label_model(single_label=choice)
        print("✅ Success: Valid instance created")
        print(f"Data: {instance.model_dump_json(indent=2)}")
    except pydantic.ValidationError as e:
        print("❌ Error: This should not happen with valid data")
        print(f"Error: {e}")
