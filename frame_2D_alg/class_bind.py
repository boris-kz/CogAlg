"""
Provide Binder object for frame_blobs_find_adj.py
"""

class Binder:
    """
    Bind different-sign adjacent composite structure of the same level.
    Parameters
    ----------
    """
    __slots__ = (
        'cs_cls',
        'adj_pairs',
        'pairs_prop',
        'pair_prop_types',
    )

    def __init__(self, cs_cls, pair_prop_types=None):
        self.cs_cls = cs_cls    # composite structure class
        self.adj_pairs = set()
        self.pairs_prop = dict()
        if pair_prop_types is not None:
            self.pair_prop_types = list(pair_prop_types)

    def bind(self, cs1, cs2):
        """
        Form a link between an adjacent pair of
        composite structures.
        """
        id1, id2 = cs1.id, cs2.id
        if id1 == id2:
            return
        elif id1 < id2:
            self.adj_pairs.add((id1, id2))
        else:
            self.adj_pairs.add((id2, id1))

    def bind_by_id(self, id1, id2):
        """
        Form a link between an adjacent pair of
        composite structures, given their ids.
        """
        if id1 == id2:
            raise ValueError("bound composite structures' ids are identical")
        elif id1 < id2:
            self.adj_pairs.add((id1, id2))
        else:
            self.adj_pairs.add((id2, id1))

    def set_adj_prop(self, cs1, cs2, prop):
        """Set the property of an adjacent pair."""
        if isinstance(prop, str):
            prop = self.pair_prop_types.index(prop)
        id1, id2 = cs1.id, cs2.id
        if id1 < id2:
            self.pairs_prop[(id1, id2)] = prop
        else:
            self.pairs_prop[(id2, id1)] = prop

    def bind_from_lower(self, sub_binder):
        """
        Form links between adjacent pairs of
        composite structures given the set of their
        element's links.
        """
        for lid1, lid2 in sub_binder.adj_pairs:
            lcs1, lcs2 = (sub_binder.cs_cls.get_cs(lid1),
                          sub_binder.cs_cls.get_cs(lid2))
            self.bind_by_id(lcs1.hid, lcs2.hid)