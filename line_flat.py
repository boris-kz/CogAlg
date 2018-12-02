import cv2
import numpy
import argparse
from time import time

# *** Utilities ****************************************************************************************

def compare( p, pri_p ):
    "Compares 2 pixel, returns d and m"
    d = p - pri_p

    return d, ( ave_d - abs( d ) )

# *** get_patterns() ***********************************************************************************

def get_patterns( current_frame ):
    "Get 1D patterns from a frame"

    global image

    for y in range( ini_y, Y ):
        pixel_ = current_frame[y, :]
        '''
        # File output *********************
        fo.write( 'pixel:\t' )
        for x in range( X ):
            fo.write( '%6d' % pixel_[ x ] )
        fo.write( '\n' )
        # *********************************
        '''
        d_ = [ 0 ] * X; m_ = [ 0 ] * X
        mP_pixel_skip_ = [ False ] * X
        dP_pixel_skip_ = [ False ] * X
        rng = 0
        Stop = False
        while not Stop :
            mP_pixel_skip_[ rng ]  = True
            dP_pixel_skip_[ rng ] = True

            rng += 1

            if( rng >= min_rng ): Stop = True

            mP_end_flag_ = [ False ] * X
            dP_end_flag_ = [ False ] * X

            D = 0; M = 0

            for x in range ( rng, X ):
                if not mP_pixel_skip_[ x ] or not dP_pixel_skip_:
                    x1 = x - rng

                    d, m = compare( pixel_[ x ], pixel_[ x1 ] )

                    d_[ x ] += d
                    d_[ x1 ] += d
                    m_[ x ] += m
                    m_[ x1 ] += m

                    if not dP_pixel_skip_[ x1 ]:
                        D += d_[ x1 ]

                    if not mP_pixel_skip_[ x1 ]:
                        M += m_[ x1 ]

            dP_ends_ = numpy.where( numpy.diff( numpy.sign( d_ ) ) )[ 0 ]
            mP_ends_ = numpy.where( numpy.diff( numpy.sign( m_ ) ) )[ 0 ]

            if rng == 2:
                for i, x in enumerate( mP_ends_):
                    image[ y, x ] = 0




            '''
            # File output *********************
            fo.write( 'rng = %d :\n' % ( rng ) )
            fo.write( 'd_:\t\t' )
            for x in range( X ):
                fo.write( '%6d' % d_[ x ] )
            fo.write( '\n' )
            fo.write( 'm_:\t\t' )
            for x in range( X ):
                fo.write( '%6d' % m_[ x ] )
            fo.write( '\n' )
            # *********************************
            '''
    return ( d_, m_ )

# *** Main *********************************************************************************************

argument_parser = argparse.ArgumentParser()
argument_parser.add_argument('-i', '--image', help='path to image file', default='./images/raccoon.jpg')
arguments = vars(argument_parser.parse_args())
image = cv2.imread(arguments['image'], 0).astype(int)

# the same image can be loaded online, without cv2:
# from scipy import misc
# f = misc.face(gray=True)  # load pix-mapped image
# f = f.astype(int)

# pattern filters are initialized here as constants, eventually adjusted by higher-level feedback:

ave_m = 10  # min d-match for inclusion in positive d_mP
ave_d = 20  # |difference| between pixels that coincides with average value of mP - redundancy to overlapping dPs
ave_M = 127  # min M for initial incremental-range comparison(t_)
ave_D = 127  # min |D| for initial incremental-derivation comparison(d_)
ini_y = 0
min_rng = 2  # fuzzy pixel comparison range, adjusted by higher-level feedback
Y, X = image.shape  # Y: frame height, X: frame width

# fo = open( 'outputs/output.txt', 'w+') # File output

start_time = time()
patterns_ = get_patterns( image )
cv2.imwrite( './images/output.jpg', image)
end_time = time() - start_time
print(end_time)

# fo.close() # File output