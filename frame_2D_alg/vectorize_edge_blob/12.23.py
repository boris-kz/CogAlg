for part, epart in zip((valt, rdnt, dect), (evalt, erdnt, edect)):
    for i in 0, 1:  # internalize externals
        part[i] += epart[i]
