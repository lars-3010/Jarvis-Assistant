"""
Fix for pairs dataset generator to respect Areas filtering.

The main issues to fix:
1. Add fallback mechanism for stratified sampling when tag groups are empty
2. Ensure pairs dataset generation respects filtered notes list
3. Update DatasetGenerator to pass filtered notes to pairs generator
"""

# Key changes needed:

# 1. In StratifiedSamplingStrategy._sample_from_groups, add check for empty groups
def _sample_from_groups_fixed(self, groups, positive_pairs, target_count):
    """Sample negative pairs from within groups."""
    pairs = []
    max_attempts = target_count * 5
    attempts = 0

    group_names = list(groups.keys())
    
    # If no groups available, return empty list
    if not group_names:
        return pairs

    while len(pairs) < target_count and attempts < max_attempts:
        # Select a group with at least 2 notes
        group_name = random.choice(group_names)
        group_notes = groups[group_name]

        if len(group_notes) >= 2:
            note_a = random.choice(group_notes)
            note_b = random.choice(group_notes)

            if note_a != note_b:
                pair = tuple(sorted([note_a, note_b]))
                if pair not in positive_pairs and pair not in pairs:
                    pairs.append(pair)

        attempts += 1

    return pairs

# 2. In _smart_negative_sampling, add fallback to random sampling
def _smart_negative_sampling_fixed(self, positive_pairs, all_notes, target_count, notes_data):
    """Implement intelligent negative example sampling with fallback."""
    try:
        # Use stratified sampling if we have note data
        if isinstance(self.sampling_strategy, StratifiedSamplingStrategy):
            # Already initialized with note_data
            pass
        elif hasattr(self.sampling_strategy, 'note_data'):
            # Update note data for stratified sampling
            self.sampling_strategy.note_data = notes_data
        else:
            # Use existing sampling strategy (e.g., random)
            pass

        negative_pairs = self.sampling_strategy.sample_negative_pairs(
            positive_pairs, all_notes, target_count
        )

        if len(negative_pairs) == 0:
            raise SamplingError(
                "Failed to generate any negative samples",
                sampling_strategy=self.sampling_strategy.get_strategy_name(),
                target_ratio=target_count / len(positive_pairs) if positive_pairs else 0
            )

        logger.info(f"Generated {len(negative_pairs)} negative pairs using {self.sampling_strategy.get_strategy_name()} strategy")
        return negative_pairs

    except Exception as e:
        logger.error(f"Negative sampling failed: {e}")
        
        # Fallback to random sampling if stratified sampling fails
        if not isinstance(self.sampling_strategy, RandomSamplingStrategy):
            logger.warning("Stratified sampling failed, falling back to random sampling")
            try:
                fallback_strategy = RandomSamplingStrategy()
                negative_pairs = fallback_strategy.sample_negative_pairs(
                    positive_pairs, all_notes, target_count
                )
                logger.info(f"Fallback random sampling generated {len(negative_pairs)} negative pairs")
                return negative_pairs
            except Exception as fallback_e:
                logger.error(f"Fallback random sampling also failed: {fallback_e}")
                raise SamplingError(f"Both stratified and random sampling failed: {e}, {fallback_e}") from e
        
        raise SamplingError(f"Negative sampling failed: {e}") from e

# 3. In DatasetGenerator, ensure filtered notes are passed to pairs generator
# This is the main fix - the pairs generator should receive the same filtered notes
# as the notes generator to ensure consistency.