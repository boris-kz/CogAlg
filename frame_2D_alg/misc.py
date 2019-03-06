from frame_2D_alg import filters
# ***************************************************** MISCELLANEOUS FUNCTIONS *****************************************
# Functions:
# -get_filters()
# ***********************************************************************************************************************
def get_filters(obj):
    " imports all variables in filters.py "
    str_ = [item for item in dir(filters) if not item.startswith("__")]
    for str in str_:
        var = getattr(filters, str)
        obj[str] = var
    # ---------- get_filters() end --------------------------------------------------------------------------------------
# ***************************************************** MISCELLANEOUS FUNCTIONS END *************************************