// Este script esta armado para ejecutar desde la linea de comando
// y sacar muchos stl como output.
// Los parametros que no son numeros los dejo fijos y el resto
// entra por linea de comando

include <polyScrewThread_r1.scad>

PI=3.141592;

hex_screw(
    d_ext,  // Outer diameter of the thread
    thr_step,  // Thread step
    step_shp_deg,  // Step shape degrees
    lg_thr_sec,  // Length of the threaded section of the screw
    1.5,  // Resolution (face at each 2mm of the perimeter)
    cntrsink,  // Countersink in both ends
    d_ext * 24/15,  // Distance between flats for the hex head
    hgt_head,  // Height of the hex head (can be zero)
    lg_nonthr_sec,  // Length of the non threaded section of the screw
    0   // Diameter for the non threaded section of the screw\
);
