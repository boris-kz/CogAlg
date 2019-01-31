import filters
# ***************************************************** MISCELLANEOUS FUNCTIONS *****************************************
# Functions:
# -get_filters()
# -tree_traverse()
# ***********************************************************************************************************************
def get_filters(obj):
    " imports all variables in filters.py "
    str_ = [item for item in dir(filters) if not item.startswith("__")]
    for str in str_:
        var = getattr(filters, str)
        obj[str] = var
    # ---------- get_filters() end --------------------------------------------------------------------------------------
def tree_traverse(tree, path):
    list = [tree[0]]
    for i, sub_path in path:
        list += tree_traverse(tree[i], sub_path)
    return list
# ***************************************************** MISCELLANEOUS FUNCTIONS END *************************************