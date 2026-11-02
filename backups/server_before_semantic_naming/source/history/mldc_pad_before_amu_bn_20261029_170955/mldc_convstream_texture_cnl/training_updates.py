"""Commit both memories only when AMP actually applied the optimizer step."""
def commit_training_memories(applied, cnl_memory, prototype_memory, unit, terms, indices, labels, cameras):
    if not applied:
        return
    if cnl_memory is not None:
        cnl_memory.update(unit, indices, labels, cameras)
    if prototype_memory is not None:
        prototype_memory.update(terms['update_features'], labels, indices)
