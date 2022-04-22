import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import sys
from matplotlib.patches import Wedge

from pprint import pprint

def test1(): # Test 360 degrees polar plot set_yticklabels
    pprint(sys.path)

    fig_test = plt.figure(figsize = (6,6))


    ax1 = fig_test.add_subplot(111,polar=True)
    ax1.set_thetamin(0)
    ax1.set_thetamax(360)
    ax1.set_yticks(np.arange(-45,46,15))
    ax1.set_yticklabels(np.arange(-45,46,15), fontsize='xx-large')
    plt.show()


def test2(): # Test 360 degrees polar plot set_tick_params

    fig_ref = plt.figure(figsize = (6, 6))
    ax2 = fig_ref.add_subplot(111,polar=True)
    ax2.set_thetamin(0)
    ax2.set_thetamax(360)
    ax2.set_yticks(np.arange(-45,46,15))
    ax2.set_yticklabels(np.arange(-45,46,15))
    ax2.yaxis.set_tick_params(labelsize='xx-large')
    plt.show()

def test3():
    # Adapted from the original github issue: https://github.com/matplotlib/matplotlib/issues/22104
    # Make fake data

    d = np.random.rand(11,18)
    azimuths = np.radians(np.arange(-42.5,42.6,5))
    radials = np.linspace(1.3,3.9,11)
    r, theta = np.meshgrid(radials, azimuths)

    # Plot the data

    fig = plt.figure(figsize=(6,6))

    ax1 = fig.add_subplot(111,polar=True)

    cont = ax1.contourf(theta, r, np.fliplr(d).T,
                       )

    ax1.set_theta_zero_location('N')
    ax1.set_thetamin(-45)
    ax1.set_thetamax(45)
    ax1.set_xticks(np.arange(np.radians(-45),
                            np.radians(46),
                            np.radians(15),
                           ))  # Less radial ticks
    ax1.set_yticks(np.arange(-45,46,15))

    ax1.set_xticklabels(np.arange(-45,46,15)[::-1],
                        fontsize='xx-large',            
                       )

    ax1.set_yticklabels(np.arange(-45,46,15)[::-1],
                        fontsize='xx-large',            
                       )
    val = ax1.get_yaxis()
    #ax1.yaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
    #ax1.yaxis.set_tick_params(labelsize='xx-large')     

    plt.show()



def test4():
    # Adapted from https://www.geeksforgeeks.org/plotting-polar-curves-in-python/
    plt.axes(projection = 'polar')
    a=3
 
    rad = np.arange(0, (2 * np.pi), 0.01)
 
    # plotting the cardioid
    for i in rad:
        r = a + (a*np.cos(i))
        plt.polar(i,r,'g.')

    plt.axes().set_yticklabels(np.arange(-45,46,45)[::-1],
                        fontsize='xx-large',            
                       )

    # display the polar plot
    plt.show()




def test5():
    # Adapted from https://stackoverflow.com/questions/30329673/how-to-set-the-axis-limit-in-a-matplotlib-plt-polar-plot
    x = np.arange(-180.0,190.0,10)
    theta = (np.pi/180.0 )*x    # in radians

    offset = 2.0

    R1 = [-0.358,-0.483,-0.479,-0.346,-0.121,0.137,0.358,0.483,0.479,0.346,0.121,\
    -0.137,-0.358,-0.483,-0.479,-0.346,-0.121,0.137,0.358,0.483,0.479,0.346,0.121,\
    -0.137,-0.358,-0.483,-0.479,-0.346,-0.121,0.137,0.358,0.483,0.479,0.346,0.121,\
    -0.137,-0.358]

    fig1 = plt.figure()
    ax1 = fig1.add_axes([0.1,0.1,0.8,0.8],polar=True)
    ax1.set_ylim(-2,2)
    ax1.set_yticks(np.arange(-2,2,0.5))
    ax1.set_yticklabels(np.arange(-2,2,0.5),
                        fontsize='xx-large',            
                       )
    ax1.yaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
    ax1.plot(theta,R1,lw=2.5)
    ax1.yaxis.set_major_formatter(mpl.ticker.FixedFormatter(np.arange(-2,2,0.5)))
    plt.show()



def test6():
    # Adapted from https://matplotlib.org/3.5.0/gallery/pie_and_polar_charts/polar_scatter.html
    # Fixing random state for reproducibility
    np.random.seed(19680801)

    # Compute areas and colors
    N = 150
    r = 2 * np.random.rand(N)
    theta = 2 * np.pi * np.random.rand(N)
    area = 200 * r**2
    colors = theta

    fig = plt.figure()
    ax = fig.add_subplot(projection='polar')
    c = ax.scatter(theta, r, c=colors, s=area, cmap='hsv', alpha=0.75)

    ax.set_thetamin(45)
    ax.set_thetamax(135)
    ax.set_yticklabels(np.arange(-2,2,0.5),
                        fontsize='xx-large',            
                       )

    plt.show()



def perp( a ) :
    ##from https://stackoverflow.com/a/3252222/2454357
    b = np.empty_like(a)
    b[0] = -a[1]
    b[1] = a[0]
    return b


def seq_intersect(a1,a2, b1,b2) :
    ##from https://stackoverflow.com/a/3252222/2454357
    da = a2-a1
    db = b2-b1
    dp = a1-b1
    dap = perp(da)
    denom = np.dot( dap, db)
    num = np.dot( dap, dp )
    return (num / denom.astype(float))*db + b1

def angle(a1, a2, b1, b2):
    ##from https://stackoverflow.com/a/16544330/2454357
    x1, y1 = a2-a1
    x2, y2 = b2-b1
    dot = x1*x2 + y1*y2      # dot product between [x1, y1] and [x2, y2]
    det = x1*y2 - y1*x2      # determinant
    return np.arctan2(det, dot)  # atan2(y, x) or atan2(sin, cos)


def draw_wedge(
    ax, r_min = 0.3, r_max = 0.5, t_min = np.pi/4, t_max = 3*np.pi/4
    ):

    ##some data
    R = np.random.rand(100)*(r_max-r_min)+r_min
    T = np.random.rand(100)*(t_max-t_min)+t_min
    ax.scatter(T,R)

    ##compute the corner points of the wedge:
    axtmin = 0

    rs = np.array([r_min,  r_max,  r_min, r_max, r_min, r_max])
    ts = np.array([axtmin, axtmin, t_min, t_min, t_max, t_max])

    ##display them in a scatter plot
    ax.scatter(ts, rs, color='r', marker='x', lw=5)

    ##from https://matplotlib.org/users/transforms_tutorial.html
    trans = ax.transData + ax.transAxes.inverted()

    ##convert to figure cordinates, for a starter
    xax, yax = trans.transform([(t,r) for t,r in zip(ts, rs)]).T

    for i,(x,y) in enumerate(zip(xax, yax)):
        ax.annotate(
            str(i), (x,y), xytext = (x+0.1, y), xycoords = 'axes fraction',
            arrowprops = dict(
                width=2,

            ),
        )


    ##compute the angles of the wedge:
    tstart = np.rad2deg(angle(*np.array((xax[[0,1,2,3]],yax[[0,1,2,3]])).T))
    tend = np.rad2deg(angle(*np.array((xax[[0,1,4,5]],yax[[0,1,4,5]])).T))

    ##the center is where the two wedge sides cross (maybe outside the axes)
    center=seq_intersect(*np.array((xax[[2,3,4,5]],yax[[2,3,4,5]])).T)

    ##compute the inner and outer radii of the wedge:
    rinner = np.sqrt((xax[1]-center[0])**2+(yax[1]-center[1])**2)
    router = np.sqrt((xax[2]-center[0])**2+(yax[2]-center[1])**2)

    wedge = Wedge(center,
                  router, tstart, tend,
                  width=router-rinner,
                  #0.6,tstart,tend,0.3,
                  transform=ax.transAxes, linestyle='--', lw=3,
                  fill=False, color='red')
    ax.add_artist(wedge)

def test7():
    fig = plt.figure(figsize=(8,4))

    ax1 = fig.add_subplot(121, projection='polar')
    ax2 = fig.add_subplot(122, projection='polar')

    ##reducing the displayed theta and r ranges in second axes:
    ax2.set_thetamin(10)
    ax2.set_thetamax(40)

    ## ax.set_rmax() does not work as one would expect -- use ax.set_ylim() instead
    ## from https://stackoverflow.com/a/9231553/2454357
    ax2.set_ylim([0.2,0.8])
    ax2.set_rorigin(-0.2)
    ax2.set_yticklabels(np.arange(-2,2,0.5),
                        fontsize='xx-large',            
                       )

    #from https://stackoverflow.com/a/41823326/2454357
    fig.canvas.draw()

    draw_wedge(ax1)
    draw_wedge(ax2, t_min=np.deg2rad(15), t_max=np.deg2rad(30))

    plt.show()

def test8(): #Minimal test case

    fig = plt.figure(figsize=(6,6))

    ax1 = fig.add_subplot(111,polar=True)


    ax1.set_thetamin(-45)
    ax1.set_thetamax(45)

    ax1.set_yticklabels(np.arange(-45,46,15)[::-1],
                        fontsize='xx-large',            
                       )
 

    plt.show()

if __name__ == "__main__":
    test1()
    test2()
    test3()
    test4()
    test5()
    test6()
    test7()
    test8()