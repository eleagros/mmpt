import os
import sys
import imagej

    
def align_imgJ(ij, path_tmp):
    # get the current working directory for inserting correctly the file paths in the macro
    path_tmp_no_start = path_tmp
    path_tmp = '"' + path_tmp
    
    
    
    macro = """
    open(""" + path_tmp + """/600nm_intensity.png");
    open(""" + path_tmp + """/to_align_x.tif");
    call("bunwarpj.bUnwarpJ_.loadLandmarks", """ + path_tmp + """/coordinates.txt");
    run("bUnwarpJ", "load=""" + path_tmp_no_start + """/coordinates.txt source_image=600nm_intensity.png target_image=to_align_x.tif registration=Accurate image_subsample_factor=0 initial_deformation=[Fine] final_deformation=[Very Fine] divergence_weight=0.1 curl_weight=0.1 landmark_weight=1.5 image_weight=0 consistency_weight=10 stop_threshold=0.01");
    saveAs("Tiff", """ + path_tmp + """/registered_img_brightfield_x.tif");
    close();
    close();
    close();
    close();
    open(""" + path_tmp + """/600nm_intensity.png");
    open(""" + path_tmp + """/to_align_y.tif");
    call("bunwarpj.bUnwarpJ_.loadLandmarks", """ + path_tmp + """/coordinates.txt");
    run("bUnwarpJ", "load=""" + path_tmp_no_start + """/coordinates.txt source_image=600nm_intensity.png target_image=to_align_y.tif registration=Accurate image_subsample_factor=0 initial_deformation=[Fine] final_deformation=[Very Fine] divergence_weight=0.1 curl_weight=0.1 landmark_weight=1.5 image_weight=0 consistency_weight=10 stop_threshold=0.01");
    saveAs("Tiff", """ + path_tmp + """/registered_img_brightfield_y.tif");
    close();
    close();
    close();
    close();
    open(""" + path_tmp + """/600nm_intensity.png");
    open(""" + path_tmp + """/550nm_intensity.png");
    call("bunwarpj.bUnwarpJ_.loadLandmarks", """ + path_tmp + """/coordinates.txt");
    run("bUnwarpJ", "load=""" + path_tmp_no_start + """/coordinates.txt source_image=600nm_intensity.png target_image=550nm_intensity.png registration=Fast image_subsample_factor=0 initial_deformation=[Fine] final_deformation=[Very Fine] divergence_weight=0.1 curl_weight=0.1 landmark_weight=1.5 image_weight=0 consistency_weight=10 stop_threshold=0.01");
    saveAs("Tiff", """ + path_tmp + """/550nm_intensity_registered.png");
    close();
    close();
    close();
    close();
    """
    ij.py.run_macro(macro)
    

current_file = sys.argv[1]
path_tmp = os.path.join(current_file, 'temp').replace('\\', '/')
ij = imagej.init(os.path.join(current_file, 'Fiji.app'), mode='interactive')
align_imgJ(ij, path_tmp)
    