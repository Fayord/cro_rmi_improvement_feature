# Final Relationship Function Flow

## Overview
The `final_relationship` function determines the final interdependency type and direction between two risks by analyzing the bidirectional analysis results (A→B and B→A).

## Priority Lists
- **Interdependency Types** (higher priority = lower index):
  1. "Causal" (highest priority)
  2. "Contingent" 
  3. "Temporal/Sequential"
  4. "Correlated"
  5. "None" (lowest priority)

- **Directions** (higher priority = lower index):
  1. "Both" (highest priority)
  2. "A → B"
  3. "B → A" 
  4. "None" (lowest priority)

## Pseudo Code

```
function final_relationship(interdependency_type_a_b, interdependency_type_b_a, direction_a_b, direction_b_a):
    
    // Step 1: Preprocess B→A direction to normalize it
    if interdependency_type_b_a in ["Causal", "Contingent", "Temporal/Sequential"]:
        if direction_b_a == "A → B":
            direction_b_a = "B → A"
        else if direction_b_a == "B → A":
            direction_b_a = "A → B"
    
    // Step 2: Check if both analyses agree on interdependency type
    if interdependency_type_a_b == interdependency_type_b_a:
        final_interdependency_type = interdependency_type_a_b
        
        // For directional types, determine final direction
        if interdependency_type_a_b in ["Causal", "Contingent", "Temporal/Sequential"]:
            if (direction_a_b, direction_b_a) forms a bidirectional pair:
                final_direction = "Both"
            else:
                // Choose the higher priority direction
                final_direction = higher_priority(direction_a_b, direction_b_a)
        else:
            // Non-directional types always have "None" direction
            final_direction = "None"
    
    // Step 3: If analyses disagree on type, choose higher priority type
    else:
        if priority(interdependency_type_a_b) > priority(interdependency_type_b_a):
            final_interdependency_type = interdependency_type_a_b
            final_direction = direction_a_b
        else:
            final_interdependency_type = interdependency_type_b_a
            final_direction = direction_b_a
    
    return final_interdependency_type, final_direction
```

## Decision Flow Table

| Condition | A→B Type | B→A Type | A→B Direction | B→A Direction | Final Type | Final Direction | Logic |
|-----------|----------|----------|---------------|---------------|------------|-----------------|-------|
| **Same Type** | Causal | Causal | A→B | B→A | Causal | Both | Bidirectional causal relationship |
| **Same Type** | Causal | Causal | A→B | A→B | Causal | A→B | Unidirectional, choose higher priority |
| **Same Type** | Correlated | Correlated | None | None | Correlated | None | Non-directional type |
| **Different Types** | Causal | Correlated | A→B | None | Causal | A→B | Choose higher priority type |
| **Different Types** | Correlated | Causal | None | B→A | Causal | B→A | Choose higher priority type |

## Key Logic Points

### 1. Direction Normalization
- When analyzing B→A, the directions are "flipped" to normalize them
- "A → B" becomes "B → A" and vice versa
- This ensures consistent comparison

### 2. Bidirectional Detection
- If A→B and B→A form opposite directions (A→B + B→A), result is "Both"
- This indicates a bidirectional relationship

### 3. Priority-Based Selection
- When types differ, the higher priority type wins
- When directions differ (same type), the higher priority direction wins
- Non-directional types (Correlated, None) always have "None" direction

### 4. Type-Specific Rules
- **Directional types** (Causal, Contingent, Temporal/Sequential): Can have A→B, B→A, Both, or None
- **Non-directional types** (Correlated, None): Always have "None" direction

## Example Scenarios

### Scenario 1: Bidirectional Causal
- A→B: Causal, A→B
- B→A: Causal, A→B (normalized to B→A)
- Result: Causal, Both

### Scenario 2: Unidirectional Causal
- A→B: Causal, A→B  
- B→A: Causal, A→B (normalized to B→A)
- Result: Causal, A→B (higher priority direction)

### Scenario 3: Type Conflict
- A→B: Causal, A→B
- B→A: Correlated, None
- Result: Causal, A→B (Causal has higher priority)

### Scenario 4: Non-directional Agreement
- A→B: Correlated, None
- B→A: Correlated, None  
- Result: Correlated, None 