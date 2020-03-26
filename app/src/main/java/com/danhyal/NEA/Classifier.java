package com.danhyal.NEA;
import android.annotation.SuppressLint;
import android.app.Activity;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.graphics.RectF;
import android.os.SystemClock;
import android.util.Log;

import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.GpuDelegate;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class Classifier {


    public static class Recognition {
    /**
     * A unique identifier for what has been recognized. Specific to the class, not the instance of
     * the object.
     */
    private final String id;

    /** Display name for the recognition. */
    private final String title;

    /**
     * A sortable score for how good the recognition is relative to others. Higher should be better.
     */
    private final Float confidence;

    /** Optional location within the source image for the location of the recognized object. */
    private RectF location;

    public Recognition(
        final String id, final String title, final Float confidence, final RectF location) {
      this.id = id;
      this.title = title;
      this.confidence = confidence;
      this.location = location;
    }

    public String getId() {
      return id;
    }

    public String getTitle() {
      return title;
    }

    public Float getConfidence() {
      return confidence;
    }

    public RectF getLocation() {
      return new RectF(location);
    }

    public void setLocation(RectF location) {
      this.location = location;
    }

    @SuppressLint("DefaultLocale")
    @Override
    public String toString() {
      String resultString = "";
      if (id != null) {
        resultString += "[" + id + "] ";
      }

      if (title != null) {
        resultString += title + " ";
      }

      if (confidence != null) {
        resultString += String.format("(%.1f%%) ", confidence * 100.0f);
      }

      if (location != null) {
        resultString += location + " ";
      }

      return resultString.trim();
    }
  }
    private int width=300;
    private int height=300;
    private final int[] intValues = new int[width * height];
    private final Interpreter.Options tfliteOptions = new Interpreter.Options();
    private MappedByteBuffer tfliteModel;
    private List<String> labels;
    protected Interpreter tflite;
    protected ByteBuffer imgData = null;
    private float[][] labelProbArray = null;
    private static final float IMAGE_MEAN = 128f;
    private static final float IMAGE_STD = 128f;
    private static final int DIM_BATCH_SIZE = 1;
    private static final int DIM_PIXEL_SIZE = 3;
    private static final int BYTES_PER_CHANNEL=4;
    private  static final int NUM_DETECTIONS=10;
    boolean isquantanized=true;
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
    ArrayList<Recognition> recognitions;
    private GpuDelegate gpuDelegate = null;
    private MappedByteBuffer loadModelFile(Activity activity) throws IOException {
        AssetFileDescriptor fileDescriptor = activity.getAssets().openFd("detect.tflite");
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
  }
  private void convertBitmapToByteBuffer(Bitmap bitmap) {
    if (imgData == null) {
      return;
    }
    imgData.rewind();
    bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
    // Convert the image to floating point.
    int pixel = 0;
    long startTime = SystemClock.uptimeMillis();
    for (int i = 0; i < width; ++i) {
      for (int j = 0; j < height; ++j) {
        final int val = intValues[pixel++];
        int pixelValue = intValues[i * height + j];
            addPixelValue(val);
      }
    }
    long endTime = SystemClock.uptimeMillis();
    Log.v("oofs","Timecost to put values into ByteBuffer: " + (endTime - startTime));
  }
    protected void addPixelValue(int pixelValue) {

        if (isquantanized){
            imgData.put((byte) ((pixelValue >> 16) & 0xFF));
            imgData.put((byte) ((pixelValue >> 8) & 0xFF));
            imgData.put((byte) (pixelValue & 0xFF));
        }else {
            imgData.putFloat((((pixelValue >> 16) & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
            imgData.putFloat((((pixelValue >> 8) & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
            imgData.putFloat(((pixelValue & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
        }

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
  public Mat recognize(Activity activity, Bitmap bitmap) throws IOException {
        outputLocations = new float[1][NUM_DETECTIONS][4];
        outputClasses = new float[1][NUM_DETECTIONS];
        outputScores = new float[1][NUM_DETECTIONS];
        numDetections = new float[1];


        labels=loadLabelList(activity);
        tfliteModel=loadModelFile(activity);

        tflite=new Interpreter(tfliteModel,tfliteOptions);
        imgData=ByteBuffer.allocate(DIM_BATCH_SIZE*width*height*DIM_PIXEL_SIZE);
        imgData.order(ByteOrder.nativeOrder());
                convertBitmapToByteBuffer(bitmap);

        Object[] inputArray = {imgData};
        Map<Integer, Object> outputMap = new HashMap<>();
        outputMap.put(0, outputLocations);
        outputMap.put(1, outputClasses);
        outputMap.put(2, outputScores);
        outputMap.put(3, numDetections);
        tflite.runForMultipleInputsOutputs(inputArray,outputMap);
         final ArrayList<Recognition> recognitions = new ArrayList<>(NUM_DETECTIONS);
      Mat drawn=new Mat();

        for (int i = 0; i < NUM_DETECTIONS; ++i) {
            if (outputScores[0][i]>0.4){
                float left=outputLocations[0][i][1] * width;
                float top=outputLocations[0][i][0] * height;
                float right=outputLocations[0][i][3] * width;
                float bottom=outputLocations[0][i][2] * height;
              final RectF detection =
                  new RectF(left,top,right,bottom);
              org.opencv.imgproc.Imgproc.rectangle(drawn,new Point(left,top),new Point(right,bottom), new Scalar(0,255,0));
              org.opencv.imgproc.Imgproc.putText(drawn,labels.get((int) outputClasses[0][i] + 1),new Point(left,top),org.opencv.imgproc.Imgproc.FONT_HERSHEY_SIMPLEX,0.5,new Scalar(255,0,0));
              // SSD Mobilenet V1 Model assumes class 0 is background class
              // in label file and class labels start from 1 to number_of_classes+1,
              // while outputClasses correspond to class index from 0 to number_of_classes
//              int labelOffset = 1;
//              recognitions.add(
//                  new Recognition(
//                      "" + i,
//                      labels.get((int) outputClasses[0][i] + labelOffset),
//                      outputScores[0][i],
//                      detection));
                }

        }
        return drawn;

  }

   protected   Classifier(Activity activity, int threads) throws IOException {

  }



    public float[][] getLabelProbArray() {
        return labelProbArray;
    }

}
