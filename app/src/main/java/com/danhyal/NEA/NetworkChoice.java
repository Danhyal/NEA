package com.danhyal.NEA;
import androidx.appcompat.app.AppCompatActivity;

import android.app.Activity;
import android.content.Context;
import android.content.Intent;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.RectF;
import android.graphics.drawable.BitmapDrawable;
import android.net.Network;
import android.os.AsyncTask;
import android.os.Bundle;
import android.os.SystemClock;
import android.util.Log;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.ProgressBar;
import android.widget.Toast;

import com.bumptech.glide.Glide;
import com.google.common.collect.Lists;
import com.google.gson.Gson;
import com.google.protobuf.ByteString;
import com.google.protobuf.Int64Value;



import org.apache.commons.codec.binary.Base64;
import org.bytedeco.ffmpeg.avutil.AVComponentDescriptor;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.PointerScope;
import org.bytedeco.javacv.AndroidFrameConverter;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.bytedeco.opencv.global.opencv_core;
import org.bytedeco.opencv.global.opencv_imgcodecs;
import org.bytedeco.opencv.global.opencv_imgproc;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Point;
import org.bytedeco.opencv.opencv_core.Scalar;
import org.bytedeco.opencv.opencv_dnn.Net;
import org.bytedeco.opencv.opencv_java;


import java.io.BufferedInputStream;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.lang.reflect.Array;
import java.lang.reflect.Type;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Timer;
import java.util.concurrent.ExecutionException;

import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfRect;
import org.opencv.core.Rect;
import org.opencv.core.Size;
import org.opencv.dnn.Dnn;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.utils.Converters;
import org.tensorflow.framework.DataType;
import org.tensorflow.framework.TensorProto;
import org.tensorflow.framework.TensorShapeProto;
//import org.tensorflow.lite.Interpreter;
//import org.tensorflow.lite.gpu.GpuDelegate;


import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;
import io.perfmark.Link;
import one.util.streamex.DoubleStreamEx;
import one.util.streamex.StreamEx;
import tensorflow.serving.Model;
import tensorflow.serving.Predict;
import tensorflow.serving.PredictionServiceGrpc;

import static android.util.Log.d;
import static org.opencv.core.Core.normalize;

public class NetworkChoice extends AppCompatActivity implements CvCameraPreview.CvCameraViewListener {
    private static Mat m1;
    static OpenCVFrameConverter.ToMat converter1 = new OpenCVFrameConverter.ToMat();
    static OpenCVFrameConverter.ToOrgOpenCvCoreMat converter2 = new OpenCVFrameConverter.ToOrgOpenCvCoreMat();
    static AndroidFrameConverter convertertobitmap=new AndroidFrameConverter();
    private static String choice;
    CvCameraPreview base;
    private static String TAG="DEBUG";
    private Timer myTimer;
     private int width=416/2;
    private int height=416/2;
    private final int[] intValues = new int[width * height];
    private MappedByteBuffer tfliteModel;
    private List<String> labels;
    List<String> Yolov3CocoClasses;

//    GpuDelegate delegate = new GpuDelegate();
//    final Interpreter.Options tfliteOptions = new Interpreter.Options().addDelegate(delegate).setAllowFp16PrecisionForFp32(true).setUseNNAPI(true);
//
//    protected Interpreter tflite;
    protected ByteBuffer imgData = null;
    private float[][] labelProbArray = null;
    private static final float IMAGE_MEAN = 127.5f;
    private static final float IMAGE_STD = 127.5f;
    private static final int DIM_BATCH_SIZE = 1;
    private static final int DIM_PIXEL_SIZE = 3;
    private static final int BYTES_PER_CHANNEL=4;
    private  static final int NUM_DETECTIONS=10;
    Net dnnnet;
    org.opencv.dnn.Net cvnet;
    boolean isquantanized=true;
    List<String> OpenImagesLabels;

    ViewGroup layout;
    org.opencv.core.Mat m2;
    private float[][][] outputLocations;
  // outputClasses: array of shape [Batchsize, NUM_DETECTIONS]
  // contains the classes of detected boxes
      private float[][] outputClasses;
      // outputScores: array of shape [Batchsize, NUM_DETECTIONS]
      // contains the scores of detected boxes
      private float[][] outputScores;
      // numDetections: array of shape [Batchsize]
      // contains the number of detected boxes
      private float[] numDetections;
    ArrayList<Classifier.Recognition> recognitions;
//    private GpuDelegate gpuDelegate = null;
    int bytesperchannel;



    static {
        Loader.load(opencv_java.class);

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



    private   String getPath(String file, Context context) {
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
            Log.i(TAG, "Failed to upload a file");
        }
        return "";
    }
    LinkedList<Object> asyncre;

    @Override
    protected void onCreate(Bundle savedInstanceState)   {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_network_choice);

         try (PointerScope scope=new PointerScope()){
             OpenImagesLabels=getcsvdata(getPath("class-descriptions-boxable.csv",this));
//             String YoloV2TinyWeights=getPath("yolov3-tiny.weights",this);
//             String YOloV2TInyCfg=getPath("yolov3-tiny.cfg",this);
//             cvnet=org.opencv.dnn.Dnn.readNetFromDarknet(YOloV2TInyCfg,YoloV2TinyWeights);
//             Yolov3CocoClasses=gettxtdata(getPath("yolov3.txt",this));

             org.opencv.core.Core.setNumThreads(8);
             org.opencv.core.Core.setUseIPP(true);
             org.bytedeco.opencv.global.opencv_core.setNumThreads(8);
             org.bytedeco.opencv.global.opencv_core.useOptimized();
             org.bytedeco.opencv.global.opencv_core.setUseOpenCL(true);
             org.bytedeco.opencv.global.opencv_core.setUseIPP(true);

            layout=findViewById(R.id.network_layout);
            base=findViewById(R.id.cameras);
            base.setCvCameraViewListener(this);

            ProgressBar progress=findViewById(R.id.progressBar1);

            Button ImgButton=findViewById(R.id.imageButton);

            progress.setVisibility(View.INVISIBLE);
            Button helpbutton=findViewById(R.id.helpbutton);
            helpbutton.setOnClickListener(new View.OnClickListener() {
                @Override
                public void onClick(View v) {
                    d(TAG,"clicked help button!");

                }
            });

            ImgButton.setOnClickListener(new View.OnClickListener() {
                @Override
                public void onClick(View v) {
                    // next steps would be to: save the frame on button click, send it off for inference then display it.
                    // after that begin new process to go back
                    ImageView imview2 = findViewById(R.id.imageView2);
                    imview2.setVisibility(View.INVISIBLE);
                    Bitmap bmp = Bitmap.createBitmap(m2.width(), m2.height(), Bitmap.Config.ARGB_8888);
                    LinkedList<Object> output=ServingGrpc(bmp);
                    org.opencv.core.Mat returned= (org.opencv.core.Mat) output.get(0);
                    System.out.println(returned.cols());



//                    LinkedList<Object> returneda=ServingGrpc(converter1.convert(converter2.convert(m2)));
                    // dumps mem
//                    saved.release();


                    d(TAG,String.format("%sx%sx%s",m2.cols(),m2.rows(),m2.channels()));

                    Toast.makeText(NetworkChoice.this,"Doing inference! /s not really ",Toast.LENGTH_LONG).show();

//                    org.opencv.core.Mat rm1= (org.opencv.core.Mat) asyncre.get(0);
//                    List<Float> detection_scores= (List<Float>) asyncre.get(1);
//                    Uftils.matToBitmap(rm1,bmp);
//                    imview2.setVisibility(View.VISIBLE);
//                    Glide.with(NetworkChoice.this).onLowMemory();
//                    Glide.with(NetworkChoice.this).load(bmp).into(imview2);




                }
            });
            scope.deallocate();


             }





    }



    public static String getChoice() {
        return choice;
    }




    private List<String> loadLabelList(Activity activity) throws IOException {
    List<String> labels = new ArrayList<String>();
    BufferedReader reader =
        new BufferedReader(new InputStreamReader(activity.getAssets().open("labels.txt")));
    String line;
    while ((line = reader.readLine()) != null) {
      labels.add(line);
    }
    reader.close();
    return labels;
  }









  private LinkedList<Object> ServingGrpc(Bitmap imgs){
        Mat img=converter1.convertToMat(convertertobitmap.convert(imgs));
        imgs.recycle();
        LinkedList<Object> Listss=new LinkedList<>();
//          String host = "192.168.1.51";
//        String Host="92.233.63.88";
        String Host="192.168.1.55";
        int Port = 8500;
        // the model's name.
        String ModelName = "resnet_openimages";
        // model's version
        long ModelVersion=1;
        final long startTime = System.currentTimeMillis();
        org.bytedeco.opencv.global.opencv_imgproc.resize(img, img, new org.bytedeco.opencv.opencv_core.Size(120, 120));
//        ImageIO.write(image, "JPEG", out);
        ByteBuffer temp = img.getByteBuffer();
        byte[] arr = new byte[temp.remaining()];
        temp.get(arr);
        int[] options= new int[]{Imgcodecs.IMWRITE_JPEG_OPTIMIZE,1};
        opencv_imgcodecs.imencode(".jpg", img, arr,options);
        // create a channel
        ManagedChannel channel = ManagedChannelBuilder.forAddress(Host, Port).usePlaintext().maxInboundMessageSize(402180070).enableFullStreamDecompression().keepAliveWithoutCalls(true).build();

        PredictionServiceGrpc.PredictionServiceBlockingStub stub = PredictionServiceGrpc.newBlockingStub(channel).withCompression("gzip");

        // create a modelspec
        Model.ModelSpec.Builder modelSpecBuilder = Model.ModelSpec.newBuilder();
        modelSpecBuilder.setName(ModelName);
        modelSpecBuilder.setVersion(Int64Value.of(ModelVersion));
        modelSpecBuilder.setSignatureName("serving_default");

        Predict.PredictRequest.Builder builder = Predict.PredictRequest.newBuilder();
        builder.setModelSpec(modelSpecBuilder);

        // create the TensorProto and request
        TensorProto.Builder tensorProtoBuilder = TensorProto.newBuilder();
        tensorProtoBuilder.setDtype(DataType.DT_STRING);
        TensorShapeProto.Builder tensorShapeBuilder = TensorShapeProto.newBuilder();
        tensorShapeBuilder.addDim(TensorShapeProto.Dim.newBuilder().setSize(1));
        tensorProtoBuilder.setTensorShape(tensorShapeBuilder.build());
        tensorProtoBuilder.addStringVal(ByteString.copyFrom(arr));
        TensorProto tp = tensorProtoBuilder.build();

        builder.putInputs("inputs", tp);

        Predict.PredictRequest request = builder.build();
        Predict.PredictResponse response = stub.withCompression("gzip").predict(request);

        Map<String, TensorProto> outputmap = response.getOutputsMap();
        int num_detections = (int) Objects.requireNonNull(outputmap.get("num_detections")).getFloatVal(0);
        List<Float> detection_classes = Objects.requireNonNull(outputmap.get("detection_classes")).getFloatValList();
        List<Float> detection_boxes_big = Objects.requireNonNull(outputmap.get("detection_boxes")).getFloatValList();
        List<List<Float>> detection_boxes = Lists.partition(detection_boxes_big, 4);
        List<Float> detection_scores= Objects.requireNonNull(outputmap.get("detection_scores")).getFloatValList();
        for (int j=0;j<num_detections;j+=1){
            double confidance=detection_scores.get(j);

            if (confidance>0.7){
                int top= (int) (detection_boxes.get(j).get(0)*img.rows());
                int left=(int)(detection_boxes.get(j).get(1)*img.cols());
                int bottom=(int)(detection_boxes.get(j).get(2)*img.rows());
                int right=(int)(detection_boxes.get(j).get(3)*img.cols());
                org.bytedeco.opencv.global.opencv_imgproc.rectangle(img,new Point(left,top),new Point(right,bottom), Scalar.GREEN);
                org.bytedeco.opencv.global.opencv_imgproc.putText(img,OpenImagesLabels.get(detection_classes.get(j).intValue()-1),new Point(left,top),1,4,Scalar.GREEN);

            }
        }
        channel.shutdown();
        final long endTime = System.currentTimeMillis();
        System.out.println("Total execution time: " + (endTime - startTime));
        // adds the image matrix and the detection scores
        Listss.add(converter2.convert(converter1.convert(img)));
        Listss.add(detection_scores);
//        detection_boxes.clear();
//        detection_classes.clear();
//        detection_boxes_big.clear();
//        img.release();
//        System.gc();
        return Listss;
    }

    public void setM1(Mat m1){
        this.m1=m1;
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

// yolo net object detection model
public org.opencv.core.Mat yolonet(org.opencv.core.Mat mat){
        String YoloV2TinyWeights=getPath("yolov3-tiny.weights",this);
             String YOloV2TInyCfg=getPath("yolov3-tiny.cfg",this);
             cvnet=org.opencv.dnn.Dnn.readNetFromDarknet(YOloV2TInyCfg,YoloV2TinyWeights);
             Yolov3CocoClasses=gettxtdata(getPath("yolov3.txt",this));
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
                    List<org.opencv.core.Rect> rects = new ArrayList<>();
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
                                rects.add(new org.opencv.core.Rect(left, top, width, height));
                            }
                        }
                    }
                    // Apply non-maximum suppression.
        try {
            float NMMSThreshold = 0.5f;
            MatOfFloat confidences = new MatOfFloat(Converters.vector_float_to_Mat(confs));
            // hopefully catches exception when the network does not detect anything.
            if (confidences.dims() == 2) {
                org.opencv.core.Rect[] boxesArray = rects.toArray(new org.opencv.core.Rect[0]);
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

    @Override
    public void onCameraViewStarted(int width, int height) {
        m1=new Mat(width, height, opencv_core.CV_8U);
        m2=new org.opencv.core.Mat(width,height,opencv_core.CV_8U);



    }


    @Override
    public void onCameraViewStopped() {
        m1.release();
        m2.release();
    }

    @Override
    public Mat onCameraFrame(Mat mat) {
            m2=converter2.convert(converter1.convert(mat));
//            LinkedList<Object> processor=ServingGrpc(mat);
//            System.out.println(processor);



        return converter1.convert(converter2.convert(m2));
    }


}
