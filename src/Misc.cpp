#include "externLib.h"
#include "solver_precision.h"
#include "Misc.h"
#include "preprocessor.h"
#include "Module_extern.h"
#include "Fluid_field_extern.h"
#include "index_cpu.h"

//======================================================================================================================================= =
//----------------------geometry related----------------------
//======================================================================================================================================= =
//*******************************set walls * *****************************
//*******************************set walls * *****************************

//===================================================================================================================================== =
//----------------------compute macroscopic varaibles from PDFs----------------------
//===================================================================================================================================== =
void compute_macro_vars_OSI() { // u,v,w,rho   (phi already known)
    T_P f[NDIR], rho, ux, uy, uz;
    int p_i, i_f;
    int p_f;

    for (p_i = 0; p_i < NGRID1; p_i++) {
        rho = 0;
        ux = 0; uy = 0; uz = 0;
        for (i_f=0; i_f < NDIR; i_f++ ){
            //p_f = i_f * NGRID1 + p_step_index(p_i + Length[i_f] - Length[i_f] * ntime );
            p_f = i_f * NGRID1 + p_step_index(p_i + Length[i_f] );
            f[i_f] = pdf[ p_f ]; 

            rho = rho + f[i_f];
            ux = ux + f[i_f] * ex[i_f] * c_LBM;            
            uy = uy + f[i_f] * ey[i_f] * c_LBM;
            uz = uz + f[i_f] * ez[i_f] * c_LBM;
        }        
        Density_rho[p_i] = rho;
        Velocity_ux[p_i] = ux /rho;
        Velocity_uy[p_i] = uy /rho;
        Velocity_uz[p_i] = uz /rho;
    }
}

//===================================================================================================================================== =
//-------------------Calculates the boundary points of the sphere at a given position---------------------
//===================================================================================================================================== =
void calculate_boundary_sphere_move(T_P Diam, T_P Coor[3], T_P Coor_old[3], int i_p) {
    T_P radius = Diam / prc(2.);
    int starti, startj ,startk;
    T_P startx, starty ,startz;

    starti = (int)(Coor[0] - radius);
    startj = (int)(Coor[1] - radius);
    startk = (int)(Coor[2] - radius);
    startx = starti - Coor[0];
    starty = startj - Coor[1];
    startz = startk - Coor[2];
    T_P rr = radius*radius;//Diam*Diam/prc(4.);        
    int intD = (int)Diam+3;

    T_P x,y,z, xx,yy,zz;
    T_P x1,y1,z1, x1x1,y1y1,z1z1;
    T_P r1r1,r2r2;
    T_P x0,y0,z0,x0x0,y0y0,z0z0,r0r0;  //the distance to the particle bebfore moving
    int qi,qj,qk;
    I_INT p_i = 0;
    //int flag_temp=0;
    I_INT refill_i = 0;
    I_INT position = 0;  //the index of the particle boundary point to the nearest flow field point
    
    for (int iz=0; iz<intD; iz++){
        for (int iy=0; iy<intD; iy++){
            for (int ix=0; ix<intD; ix++){
                x = startx + ix;  y = starty + iy;  z = startz + iz;
                xx = x*x; yy = y*y; zz = z*z;
                r1r1 = xx+yy+zz;
                if (r1r1 >= rr){
                   for (int i_f=1; i_f<NDIR; i_f++){
                        x1=x+ex[i_f];    y1=y+ey[i_f];    z1=z+ez[i_f];
                        x1x1=x1*x1; y1y1=y1*y1; z1z1=z1*z1;
                        r2r2 = x1x1+y1y1+z1z1;
                        // calculate the ratio q
                        T_P gg = (ex[i_f]*ex[i_f]+ey[i_f]*ey[i_f]+ez[i_f]*ez[i_f]);
                        T_P p1,p2p2, qq;
                        p1 = (gg + r1r1 - r2r2);
                        p2p2 = gg*gg + (r1r1-r2r2)*(r1r1-r2r2) + gg*(prc(4.)*rr - prc(2.)* (r1r1+r2r2));
                        if (p2p2 >= 0){
                            qq = (p1 - sqrt(p2p2))/(prc(2.) * gg);
                            if (qq < 1. && qq >= 0.){
                                qi = starti + ix;   qj = startj + iy;   qk = startk + iz;
                                position = p_index(qi,qj,qk);
                                Particles[i_p].BC_I[p_i] = position; //the index of the particle boundary point to the nearest flow field point
                                //Particles[i_p].BC_fOld[p_i] = pdf[ i_f * NGRID1 + p_step_index(position) ];  //the distribution of after collision in last time step
                                Particles[i_p].BC_DV[p_i] = i_f;  // the index in Particl_BC_DV is means that outside of the particle points to the inside
                                Particles[i_p].BC_q[p_i] = qq;  //The ratio of the particle boundary point to the nearest flow field point to the grid scale
                                p_i++;
                            }
                        }

                        // Calculate the refill point due to points particle moving 
                        x0 = x + (Coor[0]-Coor_old[0]);  y0 = y + (Coor[1]-Coor_old[1]);  z0 = z + (Coor[2]-Coor_old[2]);
                        x0x0 = x0*x0; y0y0 = y0*y0; z0z0 = z0*z0;
                        r0r0 = x0x0+y0y0+z0z0;   //the distance to the particle bebfore moving
                        if (r0r0 < rr){
                            int tempii;
                            T_P tempM, temp1;
                            tempM = x0*ex[1]+y0*ey[1]+z0*ez[1];
                            tempii = 1;
                            for (int ii=2; ii<NDIR; ii++){
                                temp1 = x0*ex[ii]+y0*ey[ii]+z0*ez[ii];
                                if (temp1 > tempM)
                                {
                                    tempM = temp1;
                                    tempii = ii;
                                }
                            }
                            qi = starti + ix;   qj = startj + iy;   qk = startk + iz;
                            position = p_index(qi,qj,qk);
                            Particles[i_p].Refill_Point_I[refill_i] = position;
                            Particles[i_p].Refill_Point_DV[refill_i] = tempii;
                            refill_i++;
                        }
                    }
                }
            }
        }
    }
    num_Sphere_Boundary = p_i;
    num_refill_point = refill_i;
}

void calculate_boundary_sphere(T_P Diam, T_P Coor[3], int i_p) {
    T_P radius = Diam / prc(2.);
    int starti, startj ,startk;
    T_P startx, starty ,startz;

    starti = (int)(Coor[0] - radius);
    startj = (int)(Coor[1] - radius);
    startk = (int)(Coor[2] - radius);
    startx = starti - Coor[0];
    starty = startj - Coor[1];
    startz = startk - Coor[2];
    T_P rr = radius*radius;//Diam*Diam/prc(4.);        
    int intD = (int)Diam+3;

    T_P x,y,z, xx,yy,zz;
    T_P x1,y1,z1, x1x1,y1y1,z1z1;
    T_P r1r1,r2r2;    
    int qi,qj,qk;
    I_INT p_i = 0;
    //int flag_temp=0;
    I_INT position = 0;  //the index of the particle boundary point to the nearest flow field point
    if (Diam>0){
        for (int iz=0; iz<intD; iz++){
            for (int iy=0; iy<intD; iy++){
                for (int ix=0; ix<intD; ix++){
                    x = startx + ix;  y = starty + iy;  z = startz + iz;
                    xx = x*x; yy = y*y; zz = z*z;
                    r1r1 = xx+yy+zz;
                    if (r1r1 >= rr){
                    for (int i_f=1; i_f<NDIR; i_f++){
                            x1=x+ex[i_f];    y1=y+ey[i_f];    z1=z+ez[i_f];
                            x1x1=x1*x1; y1y1=y1*y1; z1z1=z1*z1;
                            r2r2 = x1x1+y1y1+z1z1;
                            // calculate the ratio q
                            T_P gg = (ex[i_f]*ex[i_f]+ey[i_f]*ey[i_f]+ez[i_f]*ez[i_f]);
                            T_P p1,p2p2, qq;
                            p1 = (gg + r1r1 - r2r2);
                            p2p2 = gg*gg + (r1r1-r2r2)*(r1r1-r2r2) + gg*(prc(4.)*rr - prc(2.)* (r1r1+r2r2));
                            if (p2p2 >= 0){
                                qq = (p1 - sqrt(p2p2))/(prc(2.) * gg);
                                if (qq < 1. && qq >= 0.){
                                    qi = starti + ix;   qj = startj + iy;   qk = startk + iz;
                                    position = p_index(qi,qj,qk);
                                    Particles[i_p].BC_I[p_i] = position; //the index of the particle boundary point to the nearest flow field point
                                    Particles[i_p].BC_fOld[p_i] = pdf[ i_f * NGRID1 + p_step_index((Particles[i_p].BC_I[p_i])) ];//the distribution of after collision in last time step
                                    Particles[i_p].BC_DV[p_i] = i_f;  // the index in Particl_BC_DV is means that outside of the particle points to the inside
                                    Particles[i_p].BC_q[p_i] = qq;  //The ratio of the particle boundary point to the nearest flow field point to the grid scale
                                    p_i++;
                                }
                            }     
                        }
                    }
                }
            }
        }
    }
    num_Sphere_Boundary = p_i;
    num_refill_point = 0;
}

