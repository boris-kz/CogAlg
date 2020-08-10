"""
Provide AdjBinder object for frame_blobs.py
"""

class AdjBinder:
    """
    Bind different-sign adjacent clusters of the same level.
    Parameters
    ----------
    """
    __slots__ = (
        'cluster_cls',
        'adj_pairs',
        'pairs_prop',  # pair properties
        'pair_prop_types',  # pair property types. i.e internal, open...
    )

    def __init__(self, cluster_cls, pair_prop_types=None):
        self.cluster_cls = cluster_cls    # cluster class
        self.adj_pairs = set()
        self.pairs_prop = dict()
        if pair_prop_types is not None:
            self.pair_prop_types = list(pair_prop_types)

    def bind_by_id(self, id1, id2):
        """
        Form a link between an adjacent pair of
        clusters, given their ids.
        """
        if id1 == id2:
            if id1 is None:
                raise ValueError("both ids are None")
            raise ValueError("bound clusters' ids are identical")
        elif id1 < id2:  # cluster should not be bound to itself
            self.adj_pairs.add((id1, id2))
        else:
            self.adj_pairs.add((id2, id1))

    def bind(self, cs1, cs2):
        """
        Form a link between an adjacent pair of
        clusters.
        """
        id1, id2 = cs1.id, cs2.id  # get instance id
        self.bind_by_id(id1, id2)

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
        Form links between adjacent pairs of clusters
        given the set of elemental links.
        """
        for lid1, lid2 in sub_binder.adj_pairs:  # iterate over lower's adj pairs
            # get instances lower cluster by id
            lcs1, lcs2 = (sub_binder.cluster_cls.get_instance(lid1),
                          sub_binder.cluster_cls.get_instance(lid2))
            # get corresponding higher cluster ids and bind them
            self.bind_by_id(lcs1.hid, lcs2.hid)