package com.danhyal.NEA;
// fall back functions
import androidx.appcompat.app.AppCompatActivity;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.PixelFormat;
import android.graphics.PorterDuff;
import android.graphics.RectF;
import android.os.AsyncTask;
import android.os.Bundle;
import android.os.Handler;
import android.util.DisplayMetrics;
import android.util.Log;
import android.view.SurfaceHolder;
import android.view.SurfaceView;
import android.widget.Toast;



import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.indexer.UByteBufferIndexer;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.bytedeco.opencv.global.opencv_core;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.global.opencv_core;

import org.bytedeco.opencv.opencv_core.Point;
import org.bytedeco.opencv.opencv_core.CvScalar;
import org.bytedeco.opencv.opencv_core.Scalar;
import org.bytedeco.opencv.opencv_core.Scalar4f;
import org.bytedeco.opencv.opencv_core.Size;
import org.bytedeco.opencv.opencv_core.UMat;
import org.bytedeco.opencv.opencv_dnn.Net;
import org.bytedeco.opencv.opencv_java;
import org.opencv.core.Core;
import org.opencv.core.CvException;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfRect;
import org.opencv.core.Rect;
import org.opencv.dnn.Dnn;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.utils.Converters;

import org.bytedeco.opencv.global.opencv_imgproc;

import java.io.BufferedInputStream;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStream;
import java.net.HttpURLConnection;
import java.net.MalformedURLException;
import java.net.URL;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutionException;
//import org.opencv.dnn.Net;
//import org.opencv.dnn.Dnn;
import org.bytedeco.opencv.global.opencv_dnn;

import static android.util.Log.d;
import static android.util.Log.i;
import static java.nio.ByteOrder.nativeOrder;
import static org.bytedeco.opencv.global.opencv_imgproc.cvCvtColor;
import static org.bytedeco.opencv.global.opencv_imgproc.resize;
import static org.opencv.core.CvType.CV_32F;
import static org.opencv.core.CvType.CV_8U;

public class CameraAct extends AppCompatActivity implements CvCameraPreview.CvCameraViewListener {
    UMat opencl_mat;
    Mat mat2;
    Mat mat3;
    Mat mat1;
    Scalar blobscale = new Scalar();
    double[] darray = {127.5, 127.5, 127.5};
    org.opencv.core.Mat mat4;
    Net dnnnet;
    org.opencv.dnn.Net cvnet;
    int have_opencl = 0;
    final String TAG = ":::CAMERAACT:::";
    String net;

    List<String> Yolov3CocoClasses=gettxtdata(getPath("yolov3.txt",this));
    List<String> TensorCocoClasses=gettxtdata(getPath("coco.txt",this));

    // loads the opencv libs
//    TfliteClassifier classifier;
    static {
        Loader.load(opencv_java.class);

    }
    SurfaceHolder holderTransparent;
// options for model interpreter
//OverlayView trackingOverlay;

    CvCameraPreview javacvviewbase;
    OpenCVFrameConverter.ToMat converter1 = new OpenCVFrameConverter.ToMat();
    OpenCVFrameConverter.ToOrgOpenCvCoreMat converter2 = new OpenCVFrameConverter.ToOrgOpenCvCoreMat();

    @Override
    protected void onCreate(Bundle savedInstanceState) {

        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_camera);

        javacvviewbase = (CvCameraPreview) findViewById(R.id.cameraact);
        javacvviewbase.setCvCameraViewListener(this);

//        SurfaceView transparentView = (SurfaceView)findViewById(R.id.TransparentView);
//
//       holderTransparent = transparentView.getHolder();
//        holderTransparent.setFormat(PixelFormat.TRANSPARENT);
//        holderTransparent.setType(SurfaceHolder.SURFACE_TYPE_PUSH_BUFFERS);

    }

private void DrawFocusRect(float RectLeft, float RectTop, float RectRight, float RectBottom, int color)
{

    Canvas canvas = holderTransparent.lockCanvas();
    canvas.drawColor(0, PorterDuff.Mode.CLEAR);
    //border's properties
    Paint paint = new Paint();
    paint.setStyle(Paint.Style.STROKE);
    paint.setColor(color);
    paint.setStrokeWidth(3);
    canvas.drawRect(RectLeft, RectTop, RectRight, RectBottom, paint);


    holderTransparent.unlockCanvasAndPost(canvas);
}
    @Override
    public void onCameraViewStarted(int width, int height) {
        opencl_mat = new UMat(width, height, opencv_core.CV_8U);
        mat1=new Mat(width, height, opencv_core.CV_8U);
        mat2 = new Mat(width, height, opencv_core.CV_8U);
        mat3 = new Mat(width, height, opencv_core.CV_8U);
        mat4 =new org.opencv.core.Mat(width, height, opencv_core.CV_32F);





//        String weights = getPath("frozen_inference_graph.pb", this);
//        String config = getPath("ssdlite_graph.pbtxt", this);
          String YoloV2TinyWeights=getPath("yolov3-tiny.weights",this);
          String YOloV2TInyCfg=getPath("yolov3-tiny.cfg",this);
//        String proto=getPath("deploy.prototxt",this);
//        String caffe=getPath("MobileNetSSD_deploy.caffemodel",this);
//        dnnnet = opencv_dnn.readNetFromCaffe(proto,caffe);
//        dnnnet = opencv_dnn.readNetFromTensorflow(weights, config);

//        dnnnet=opencv_dnn.readNetFromDarknet(YoloV2TinyWeights,YOloV2TInyCfg);
        cvnet=org.opencv.dnn.Dnn.readNetFromDarknet(YOloV2TInyCfg,YoloV2TinyWeights);
//        try {
//            classifier=new TfliteClassifier(this);
//        } catch (IOExcept yion e) {
//            e.printStackTrace();
//        }

        Log.i(TAG, "Network loaded successfully");
        Toast.makeText(this,"Network Initialized",Toast.LENGTH_LONG).show();
    }

    @Override
    public void onCameraViewStopped() {
        opencl_mat.release();
        mat2.release();
        mat3.release();
        mat1.release();
        mat4.release();


    }

     public  List<String> getcsvdata(String filename){
        List<String> tarray=new ArrayList<>();
        String line="";
        String csvpath=getPath(filename,this);
        try (BufferedReader br = new BufferedReader(new FileReader(csvpath))) {
            while ((line = br.readLine()) != null) {
                String[] elements = line.split(",");
                tarray.add(elements[1]);
            }

        } catch (IOException e) {
            e.printStackTrace();
        }
        return tarray;
    }

    // load items from assets folder
    public  String getPath(String file, Context context) {
        AssetManager assetManager = context.getAssets();
        BufferedInputStream inputStream = null;
        try {
            // Read data from assets.
            inputStream = new BufferedInputStream(assetManager.open(file));
            byte[] data = new byte[inputStream.available()];
            inputStream.read(data);
            inputStream.close();
            // Create copy file in storage.
            File outFile = new File(context.getFilesDir(), file);
            FileOutputStream os = new FileOutputStream(outFile);
            os.write(data);
            os.close();
            // Return a path to file which may be read in common way.
            return outFile.getAbsolutePath();
        } catch (IOException ex) {
            Log.i("oof", "Failed to upload a file");
        }
        return "";
    }
     private static List<String> getOutputNames(org.opencv.dnn.Net net) {
        // layer names
        List<String> names = new ArrayList<>();
        // retrieves output layers from loaded cfg
        List<Integer> outLayers = net.getUnconnectedOutLayers().toList();
        List<String> layersNames = net.getLayerNames();

//        outLayers.forEach((item) -> names.add(layersNames.get(item - 1)));
        for(int item:outLayers){
            names.add(layersNames.get(item-1));
        }
        return names;
    }
    org.opencv.core.Mat mat5 =new org.opencv.core.Mat();
List<org.opencv.core.Mat> objects;
    @Override
    public Mat onCameraFrame(Mat mat) {

        blobscale.put(darray);
//        dnnnet.setPreferableTarget(opencv_dnn.F);
//        String NetChoice=NetworkChoice.getChoice();
        String NetChoice="yolo";
        if (NetChoice.equals("yolo")){
            if (org.bytedeco.opencv.global.opencv_core.haveOpenCL()){
                cvnet.setPreferableTarget(Dnn.DNN_TARGET_OPENCL);
            }
//            Toast.makeText(this,"Loading Yolov3 Network",Toast.LENGTH_LONG).show();
               mat5=converter2.convert(converter1.convert(mat));
            org.opencv.imgproc.Imgproc.resize(mat5,mat5,new org.opencv.core.Size(416,416));
            mat5=processyolo(mat5);

//                yoloasync yoloasync=new yoloasync();
//                yoloasync.execute(mat5);
//            yoloasync yoloasync=new yoloasync(getApplicationContext(), new OnEventListener<List<org.opencv.core.Mat>>() {
//
//
//
//
//        @Override
//        public void onSuccess(List<org.opencv.core.Mat> object) {
//            objects=object;
//
//
//
//                d(TAG,"sucscess!");
//        }
//
//        @Override
//        public void onFailure(Exception e) {
//
//        }
//    });
//    yoloasync.execute(mat5);




//            try {
//                mat5=bigoof(mat5);
//
//            }catch (Exception e){e.printStackTrace();}


        }else if (NetChoice.equals("tensor")){
            if (org.bytedeco.opencv.global.opencv_core.haveOpenCL()){
                mat2 = MobileNetCL(mat).getMat(opencv_core.ACCESS_RW);

            }else {
                mat2=MobileNet(mat);
            }

        }

//        org.bytedeco.opencv.global.opencv_imgproc.resize(mat2,mat2,new Size(GetDimensions()[1],GetDimensions()[0]),0.5,0.5, opencv_imgproc.INTER_CUBIC);
        return converter1.convert(converter2.convert(mat5));
    }

    public Mat MobileNet(org.bytedeco.opencv.opencv_core.Mat mat){
            final int INFERENCE_WIDTH = 288;
            final int INFERENCE_HEIGHT = 288;
            final float THRESHOLD = 0.6f;
            org.bytedeco.opencv.global.opencv_imgproc.resize(mat, mat, new Size(INFERENCE_WIDTH, INFERENCE_HEIGHT));
            org.bytedeco.opencv.global.opencv_imgproc.cvtColor(mat, mat, opencv_imgproc.COLOR_RGB2BGR);
            Mat cvblob = opencv_dnn.blobFromImage(mat, 1, new Size(INFERENCE_WIDTH, INFERENCE_HEIGHT), null, false, false, opencv_core.CV_8U);
            dnnnet.setInput(cvblob);
            Mat cvout = dnnnet.forward();
            int ndims = cvout.dims();
            int lastdim = cvout.size(ndims - 1);
//            d(TAG, (String.format("TAG:::::(%d) (%d)", ndims, lastdim)));
            Mat networkdetections = cvout.reshape(1, (int) cvout.total() / 7);
            for (int i = 0; i < networkdetections.rows(); i += 1) {
                double confidance = converter2.convert(converter1.convert(networkdetections)).get(i, 2)[0];
//                    d(TAG,String.valueOf(confidance));
                if (confidance > THRESHOLD) {
                    int classId = (int) converter2.convert(converter1.convert(networkdetections)).get(i, 1)[0];
                    int left = (int) (converter2.convert(converter1.convert(networkdetections)).get(i, 3)[0] * mat.cols());
                    int top = (int) (converter2.convert(converter1.convert(networkdetections)).get(i, 4)[0] * mat.rows());
                    int right = (int) (converter2.convert(converter1.convert(networkdetections)).get(i, 5)[0] * mat.cols());
                    int bottom = (int) (converter2.convert(converter1.convert(networkdetections)).get(i, 6)[0] * mat.rows());
                    org.bytedeco.opencv.global.opencv_imgproc.rectangle(mat, new Point(left, top), new Point(right, bottom), Scalar.GREEN);
                    org.bytedeco.opencv.global.opencv_imgproc.putText(mat,TensorCocoClasses.get(classId),new Point(left,top), opencv_imgproc.FONT_HERSHEY_SIMPLEX,0.5,Scalar.BLACK);
                }
            }
            return mat;

    }
         public  List<String> gettxtdata(String filename){
        List<String> tarray=new ArrayList<>();
        String line="";
        String csvpath=getPath(filename,this);
        try (BufferedReader br = new BufferedReader(new FileReader(csvpath))) {
            while ((line = br.readLine()) != null) {
                tarray.add(line);
            }

        } catch (IOException e) {
            e.printStackTrace();
        }
        return tarray;
    }
    // uses the opencl backend when the lib is found
    public UMat MobileNetCL(org.bytedeco.opencv.opencv_core.Mat mat){
            dnnnet.setPreferableBackend(opencv_dnn.DNN_BACKEND_OPENCV);
            dnnnet.setPreferableTarget(opencv_dnn.DNN_TARGET_OPENCL);
            final float THRESHOLD = 0.6f;
            final int INFERENCE_WIDTH = 288;
            final int INFERENCE_HEIGHT = 288;
            opencl_mat = mat.getUMat(opencv_core.ACCESS_RW);
            org.bytedeco.opencv.global.opencv_imgproc.resize(opencl_mat, opencl_mat, new Size(INFERENCE_WIDTH, INFERENCE_HEIGHT));
            org.bytedeco.opencv.global.opencv_imgproc.cvtColor(opencl_mat, opencl_mat, opencv_imgproc.COLOR_RGB2BGR);
            UMat cvblob_cl = opencv_dnn.blobFromImage(opencl_mat, 1, new Size(INFERENCE_WIDTH, INFERENCE_HEIGHT), null, false, false, opencv_core.CV_8U).getUMat(opencv_core.ACCESS_RW);
            dnnnet.setInput(cvblob_cl);
            Mat cvout_cl = dnnnet.forward();
            Mat networkdections_cl = cvout_cl.reshape(1, (int) cvout_cl.total() / 7);
            for (int i = 0; i < networkdections_cl.rows(); i += 1) {
                double confidence_cl = converter2.convert(converter1.convert(networkdections_cl)).get(i, 2)[0];
                if (confidence_cl > THRESHOLD) {
                    int classId = (int) converter2.convert(converter1.convert(networkdections_cl)).get(i, 1)[0];
                    int left = (int) (converter2.convert(converter1.convert(networkdections_cl)).get(i, 3)[0] * opencl_mat.cols());
                    int top = (int) (converter2.convert(converter1.convert(networkdections_cl)).get(i, 4)[0] * opencl_mat.rows());
                    int right = (int) (converter2.convert(converter1.convert(networkdections_cl)).get(i, 5)[0] * opencl_mat.cols());
                    int bottom = (int) (converter2.convert(converter1.convert(networkdections_cl)).get(i, 6)[0] * opencl_mat.rows());
                    org.bytedeco.opencv.global.opencv_imgproc.rectangle(opencl_mat, new Point(left, top), new Point(right, bottom), Scalar.GREEN);
                    org.bytedeco.opencv.global.opencv_imgproc.putText(opencl_mat,TensorCocoClasses.get(classId),new Point(left,top),opencv_imgproc.FONT_HERSHEY_SIMPLEX,0.5,Scalar.BLACK);
                }
            }
            return opencl_mat;

    }
// yolo net object detection model
public org.opencv.core.Mat yolonet(org.opencv.core.Mat mat){
           List<org.opencv.core.Mat> result = new ArrayList<>();

           Imgproc.resize(mat, mat, new org.opencv.core.Size(416, 416));
           Imgproc.cvtColor(mat, mat, opencv_imgproc.COLOR_RGB2BGR);

           org.opencv.core.Mat cvblob = Dnn.blobFromImage(mat, 1.0 / 255.0, new org.opencv.core.Size(416, 416), new org.opencv.core.Scalar(0, 0, 0), false, true, opencv_core.CV_32F);
           cvnet.setInput(cvblob);
           List<String> outnames;
           outnames = getOutputNames(cvnet);
           cvnet.forward(result, outnames);
         float confThreshold = 0.7f;
                    List<Integer> clsIds = new ArrayList<>();
                    List<Float> confs = new ArrayList<>();
                    List<Rect> rects = new ArrayList<>();
                    for (int i = 0; i < result.size(); ++i)
                    {
                        // each row is a candidate detection,
                        org.opencv.core.Mat level = result.get(i);

                        for (int j = 0; j < level.rows(); ++j)
                        {
                            org.opencv.core.Mat row = level.row(j);
                            org.opencv.core.Mat scores = row.colRange(5, level.cols());
                            Core.MinMaxLocResult mm = Core.minMaxLoc(scores);
                            float NetConfidence = (float)mm.maxVal;
                            org.opencv.core.Point classIdPoint = mm.maxLoc;
                            if (NetConfidence > confThreshold)
                            {
                                int centerX = (int)(row.get(0,0)[0] * mat.cols());
                                int centerY = (int)(row.get(0,1)[0] * mat.rows());
                                int width   = (int)(row.get(0,2)[0] * mat.cols());
                                int height  = (int)(row.get(0,3)[0] * mat.rows());
                                int left    = centerX - width  / 2;
                                int top     = centerY - height / 2;

                                clsIds.add((int)classIdPoint.x);
                                confs.add((float)NetConfidence);
                                rects.add(new Rect(left, top, width, height));
                            }
                        }
                    }
                    // Apply non-maximum suppression.
        try {
            float NMMSThreshold = 0.5f;
            MatOfFloat confidences = new MatOfFloat(Converters.vector_float_to_Mat(confs));
            // hopefully catches exception when the network does not detect anything.
            if (confidences.dims() == 2) {
                Rect[] boxesArray = rects.toArray(new Rect[0]);
                MatOfRect boxes = new MatOfRect(boxesArray);
                MatOfInt indices = new MatOfInt();

                Dnn.NMSBoxes(boxes, confidences, confThreshold, NMMSThreshold, indices);

                // Draw result boxes:
                int[] ind = indices.toArray();
                for (int i = 0; i < ind.length; ++i) {

                    int idx = ind[i];
                    Rect box = boxesArray[idx];
                    d(TAG, String.valueOf(box));
                    // draws rectangle on the frame bases on the confidance
                    Imgproc.rectangle(mat, box.tl(), box.br(), new org.opencv.core.Scalar(0, 255, 0), 1);
                    // displays classes on above detection box.
                    Imgproc.putText(mat, Yolov3CocoClasses.get(clsIds.get(idx)), box.tl(), opencv_imgproc.FONT_HERSHEY_SIMPLEX, 0.5, new org.opencv.core.Scalar(0, 0, 0), 1);

                }
            }

      }catch (Exception e){e.printStackTrace();}
        return mat;
}


Handler handler;
synchronized void runInBackground(final Runnable r) {
    if (handler != null) {
      handler.post(r);
    }
  }


  public org.opencv.core.Mat t1(org.opencv.core.Mat mat) throws ExecutionException, InterruptedException {
              List<org.opencv.core.Mat> result = new ArrayList<>();

           Imgproc.resize(mat, mat, new org.opencv.core.Size(416, 416));
           Imgproc.cvtColor(mat, mat, opencv_imgproc.COLOR_RGB2BGR);

           org.opencv.core.Mat cvblob = Dnn.blobFromImage(mat, 1.0 / 255.0, new org.opencv.core.Size(416, 416), new org.opencv.core.Scalar(0, 0, 0), false, true, opencv_core.CV_32F);
           cvnet.setInput(cvblob);
           List<String> outnames;
           outnames = getOutputNames(cvnet);
           cvnet.forward(result, outnames);

    return mat5;
  }



  public interface OnEventListener<T> {
    public void onSuccess(T object);
    public void onFailure(Exception e);
}





  public class yoloasync extends  AsyncTask<org.opencv.core.Mat,Void, List<org.opencv.core.Mat>>{
          List<org.opencv.core.Mat> result = new ArrayList<>();
          private OnEventListener<List<org.opencv.core.Mat>> mCallBack;
          private Context mContext;
          public Exception mException;


          public yoloasync(Context context,OnEventListener callback){
          mCallBack=callback;
          mContext=context;
          }


          Rect box;
      @Override
      protected List<org.opencv.core.Mat> doInBackground(org.opencv.core.Mat... mats) {
           org.opencv.core.Mat mat=mats[0];
           Imgproc.resize(mat, mat, new org.opencv.core.Size(416, 416));
           Imgproc.cvtColor(mat, mat, opencv_imgproc.COLOR_RGB2BGR);

           org.opencv.core.Mat cvblob = Dnn.blobFromImage(mat, 1.0 / 255.0, new org.opencv.core.Size(416, 416), new org.opencv.core.Scalar(0, 0, 0), false, true, opencv_core.CV_32F);
           cvnet.setInput(cvblob);
           List<String> outnames;
           outnames = getOutputNames(cvnet);
           cvnet.forward(result, outnames);

        return result;
      }

      @Override
      protected void onPostExecute(List<org.opencv.core.Mat> mat) {
          if (mCallBack != null) {
            if (mException == null) {
                mCallBack.onSuccess(mat);
            } else {
                mCallBack.onFailure(mException);
            }
        }
      }
  }
    public org.opencv.core.Mat processyolo(org.opencv.core.Mat mat){
                      List<org.opencv.core.Mat> result = new ArrayList<>();
                      org.bytedeco.javacv.Parallel.run(new Runnable() {
                          @Override
                          public void run() {
                              Imgproc.resize(mat, mat, new org.opencv.core.Size(416, 416));
                                Imgproc.cvtColor(mat, mat, opencv_imgproc.COLOR_RGB2BGR);

                                org.opencv.core.Mat cvblob = Dnn.blobFromImage(mat, 1.0 / 255.0, new org.opencv.core.Size(416, 416), new org.opencv.core.Scalar(0, 0, 0), false, true, opencv_core.CV_32F);
                                cvnet.setInput(cvblob);
                                List<String> outnames;
                                outnames = getOutputNames(cvnet);
                                cvnet.forward(result, outnames);
                                d(TAG,String.valueOf(result.size()));
                          }
                      });



        return mat;
    }
    // fix for gc dealloc error
    public static Mat toMat(UMat u) {
    final Mat clone = new Mat(u.rows(), u.cols(), u.type());
    u.copyTo(clone);
    return clone;
    }
    // same but for opencl mats
    public static UMat toUMat(Mat u) {
    final UMat clone = new UMat(u.rows(), u.cols(), u.type());
    u.copyTo(clone);
    return clone;
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
