from plotting_methods import *


def main(fname, which_year, peak=False):

    #ds full
    ds = xr.open_dataset(fname,  chunks={'time':1,'x':1000,'y':1000})
    ds.close()


    if peak:
        #~~~~~~~~~~~ds peak
        dpeak = ds.isel(time=[0,7,18,30,42,49]) 
        dpeak.close()

        ds = dpeak
        ds.close()
        ds
        temp_title = 'Snow distribution nearest peak SWE flight date'
        make_my_super_plot(ds,title=temp_title, color_scheme='q')

    else:
        #~~~~~~~~~~~~~~ds small
        dsmall = ds.sel(time='{}'.format(which_year))
        dsmall.close()

        ds = dsmall
        ds.close()
        temp_title = '{} snow depth distribution, binx=1 [cm]'.format(which_year)
        make_my_super_plot(ds,title=temp_title, color_scheme='s', xlim=[0,600])


if __name__ == "__main__":
    # Look up argparse
    # python main_template.py -f fname -y 2013 --peak

#     for yr in range(2013,2019):
#         main('~/Documents/projects/thesis/results/output/compiled_SUPERsnow.nc',yr)


    main('~/Documents/projects/thesis/results/output/compiled_SUPERsnow.nc',2013)
