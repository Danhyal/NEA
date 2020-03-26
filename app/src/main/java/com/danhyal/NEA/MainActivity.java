package com.danhyal.NEA;
import android.Manifest;
import android.app.Activity;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Camera;
import android.os.Bundle;
import android.speech.tts.TextToSpeech;
import android.util.DisplayMetrics;
import android.view.SurfaceView;
import android.view.View;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ProgressBar;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import org.bytedeco.opencv.global.opencv_core;
import org.opencv.android.BaseLoaderCallback;
//import org.opencv.android.CameraBridgeViewBase;
//import org.opencv.android.JavaCamera2View;
//import org.opencv.android.OpenCVLoader;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.MatOp;
//import org.opencv.core.CvType;
//import org.opencv.imgproc.Imgproc;


import java.util.Locale;

import static android.util.Log.d;

public class MainActivity extends AppCompatActivity {

    Mat mat1,resized,mat3;
    int CameraRequestCode=1001;
    private static final int CAMERA_REQUEST = 50;
    public static boolean FlashStatus=false;
    Camera cam;
//    CameraBridgeViewBase opencvviewbase;
//    CvCameraPreview javacvviewbase;

    public int CameraIds=0;
    private boolean btnclickmeclicked = false;
    public int START_CAM=0;
    public static final String CAMERA_FRONT = "1";
    public static final String CAMERA_BACK = "0";
    public boolean PermissionGranted=false;
    private String cameraId = CAMERA_BACK;
    private boolean isFlashSupported;
    private boolean isTorchOn;
    TextToSpeech ttobj;
    public ContextCompat getcontext;

    static{
        System.setProperty("org.bytedeco.javacpp.maxphysicalbytes", "0");
        System.setProperty("org.bytedeco.javacpp.maxbytes", "0");
    }
    @Override
    protected void onCreate(Bundle savedInstanceState) {

        org.bytedeco.opencv.global.opencv_core.setUseOpenCL(true);

        super.onCreate(savedInstanceState);;

        ttobj=new TextToSpeech(getApplicationContext(), new TextToSpeech.OnInitListener() {
           @Override
           public void onInit(int status) {
               if (status==TextToSpeech.SUCCESS){
                   ttobj.setLanguage(Locale.ENGLISH);

               }
           }
        });

        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_HARDWARE_ACCELERATED);

        setContentView(R.layout.activity_main);

        Button PermissionsButton=(Button) findViewById(R.id.PermissionsButton);
        ProgressBar progress=findViewById(R.id.progressmainact);
        ttobj.speak("Click to grant permissions",TextToSpeech.QUEUE_FLUSH,null);
        progress.setVisibility(View.INVISIBLE);
        View.OnClickListener PermissionButtonClick=((View v)->{

            ActivityCompat.requestPermissions(MainActivity.this, new String[]{Manifest.permission.CAMERA}, CameraRequestCode);
            ActivityCompat.requestPermissions(MainActivity.this, new String[]{Manifest.permission.READ_EXTERNAL_STORAGE}, 3);
                Toast.makeText(this,"clicked button",Toast.LENGTH_SHORT);
                d("OOF","clicked button!");
//                progress.setVisibility(View.VISIBLE);

            if (ContextCompat.checkSelfPermission(MainActivity.this,Manifest.permission.CAMERA)==PackageManager.PERMISSION_GRANTED){
                        Intent intent = new Intent(MainActivity.this, com.danhyal.NEA.NetworkChoice.class);
                        startActivity(intent);
                    }


        });
        PermissionsButton.setOnClickListener(PermissionButtonClick);

    }
    public void SetPermissions() {
                 // Here, thisActivity is the current activity
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {

            // Permission is not granted
            // Should we show an explanation?
            if (ActivityCompat.shouldShowRequestPermissionRationale(this, Manifest.permission.CAMERA)) {
                // Show an explanation to the user *asynchronously* -- don't block
                // this thread waiting for the user's response! After the user
                // sees the explanation, try again to request the permission.
            } else {
                // No explanation needed; request the permission
                ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.CAMERA}, CameraRequestCode);


                // MY_PERMISSIONS_REQUEST_READ_CONTACTS is an
                // app-defined int constant. The callback method gets the
                // result of the request.
            }
        } else {
            PermissionGranted=true;
            // Permission has already been granted
        }
    }
    public int[] GetDimensions(){
        DisplayMetrics displayMetrics = new DisplayMetrics();
        getWindowManager().getDefaultDisplay().getMetrics(displayMetrics);
        int height = displayMetrics.heightPixels;
        int width = displayMetrics.widthPixels;
        int[] heightandwidth=new int[]{height,width};
        return heightandwidth;
    }
}
