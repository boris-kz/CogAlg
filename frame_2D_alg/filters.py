import math as __math__
ave = 20            # g value that coincides with average match: gP filter
div_ave = 1023      # filter for div_comp(L) -> rL, summed vars scaling
flip_ave = 10000    # cost of form_P and deeper?
ave_rate = 0.25     # match rate: ave_match_between_ds / ave_match_between_ps, init at 1/4: I / M (~2) * I / D (~2)
angle_coef = 128 / __math__.pi  # coef to convert radian to 256 degrees
A_cost = 1000
a_cost = 15
input_path = './../images/raccoon_eye.jpg'