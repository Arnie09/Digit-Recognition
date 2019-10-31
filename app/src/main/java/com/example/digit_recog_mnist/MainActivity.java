package com.example.digit_recog_mnist;

import androidx.appcompat.app.AppCompatActivity;

import android.app.Activity;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.os.Bundle;
import android.util.Log;
import android.view.Surface;
import android.view.SurfaceView;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.JavaCameraView;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;
import org.tensorflow.lite.Interpreter;

import java.io.FileInputStream;
import java.io.IOException;
import java.lang.reflect.Array;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;

public class MainActivity extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2 {

    CameraBridgeViewBase cameraBridgeViewBase;
    BaseLoaderCallback baseLoaderCallback;
    Mat mat1,mat2,mat3;
    Mat imageToanalyse;
    ImageView myImageView;
    Bitmap img;
    protected Interpreter tflite;
    TextView prediction;
    Button analyse;

    private static int digit = -1;
    private static float  prob = 0.0f;

    private static final int DIM_BATCH_SIZE = 1;
    private static final int DIM_PIXEL_SIZE =1;
    private static final int  DIM_HEIGHT =28;
    private static final int DIM_WIDTH = 28;
    private static final int BYTES =4;
    protected ByteBuffer imgData = null;
    private float[][] ProbArray = null;
    private String TAG = "MNIST";
    public static final String ModelFile = "cnn_model.tflite";

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        prediction = findViewById(R.id.prediction);
        analyse = findViewById(R.id.Capture);
        cameraBridgeViewBase = (JavaCameraView)findViewById(R.id.myCameraView);
        cameraBridgeViewBase.setVisibility(SurfaceView.VISIBLE);
        cameraBridgeViewBase.setCvCameraViewListener(this);
        myImageView = findViewById(R.id.ImageView);


        //initialise the tflite interpreter
        try {
            tflite = new Interpreter(loadModelFile(MainActivity.this));
        }
        catch (Exception e){
            Log.i("Interference",e.getMessage());
        }

        //initialise the input of the neural net
        imgData = ByteBuffer.allocateDirect(DIM_BATCH_SIZE * DIM_HEIGHT * DIM_WIDTH * DIM_PIXEL_SIZE * BYTES);
        imgData.order(ByteOrder.nativeOrder());
        ProbArray = new float[1][10];


        baseLoaderCallback = new BaseLoaderCallback(this) {
            @Override
            public void onManagerConnected(int status) {
                super.onManagerConnected(status);
                switch (status){

                    case BaseLoaderCallback.SUCCESS:
                        cameraBridgeViewBase.enableView();
                        break;
                    default:
                        super.onManagerConnected(status);
                        break;
                }
            }
        };

        analyse.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                if(tflite!=null) {
                    convertMattoTfLiteInput(imageToanalyse);
                    runInference();
                }else{
                    Log.i("Interference","tflite is null");
                }
            }
        });

    }

    //link the tflite model with the interpreter
    private MappedByteBuffer loadModelFile(Activity activity) throws IOException {
        AssetFileDescriptor fileDescriptor = activity.getAssets().openFd(ModelFile);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    @Override
    public void onCameraViewStarted(int width, int height) {
        mat1 = new Mat(width,height, CvType.CV_8UC4);
        mat2 = new Mat();
        mat3 = new Mat();
        imageToanalyse = new Mat(28,28,CvType.CV_8UC4);
    }

    @Override
    public void onCameraViewStopped() {
        mat1.release();
        mat2.release();
        mat3.release();
        imageToanalyse.release();
    }

    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {

        mat1=inputFrame.rgba();
        Core.transpose(mat1,mat1);
        Core.flip(mat1,mat1,1);


        int top = mat1.rows()/2 - 140;
        int left = mat1.cols() / 2 - 140;
        int height = 140*2;
        int width = 140*2;

        ///prepocess frame
        Mat gray = inputFrame.gray();
        //draw cropped region
        Imgproc.rectangle(mat1, new Point(mat1.cols()/2 - 150, mat1.rows() / 2 - 150), new Point(mat1.cols() / 2 + 150, mat1.rows() / 2 + 150), new Scalar(255,255,255),5);
        //crop frame
        Mat graytemp = gray.submat(top, top + height, left, left + width);
        //blur the cropped frame to remove noise
        Imgproc.GaussianBlur(graytemp, graytemp, new org.opencv.core.Size(7,7),2 , 2);
        //convert gray frame to binary using apadative thresold
        Imgproc.adaptiveThreshold(graytemp, mat2, 255, Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C, Imgproc.THRESH_BINARY_INV, 5, 5);
        Mat element1 = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new org.opencv.core.Size(9,9));
        //dilate the frame
        Imgproc.dilate(mat2, mat2, element1);

        Mat element = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new org.opencv.core.Size(3,3));
        //erode the frame
        Imgproc.erode(mat2, mat2, element);

        Imgproc.resize(mat2, imageToanalyse, new org.opencv.core.Size(28,28));///CNN input
        Core.transpose(imageToanalyse,imageToanalyse);
        Core.flip(imageToanalyse,imageToanalyse,1);

        //Log.i("Interference","We are here");


        //classifier.classifyMat(CNN_input);
        //Imgproc.putText(mRgba, "Digit: "+classifier.getdigit()+ " Prob: "+classifier.getProb(), new Point(top, left), 3, 3, new Scalar(255, 0, 0, 255), 2);


        img = Bitmap.createBitmap(imageToanalyse.cols(), imageToanalyse.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(imageToanalyse, img);

        graytemp.release();

        runOnUiThread(new Runnable() {
            @Override
            public void run() {
                if( myImageView != null )
                    myImageView.setImageBitmap(img);
            }
        });

        return mat1;
    }

    private void runInference() {
        Log.i("Interference","Here");
        if(imgData != null)
            tflite.run(imgData, ProbArray);
        Log.e("Interference", "Inference done "+maxProbIndex(ProbArray[0]));
    }

    private int maxProbIndex(float[] probs) {
        int maxIndex = -1;
        float maxProb = 0.0f;
        for (int i = 0; i < probs.length; i++) {
            if (probs[i] > maxProb) {
                maxProb = probs[i];
                maxIndex = i;
            }
        }
        prob = maxProb;
        digit = maxIndex;
        return maxIndex;
    }

    //convert opencv mat to tensorflowlite input
    private void convertMattoTfLiteInput(Mat mat)
    {
        imgData.rewind();
        int pixel = 0;
        for (int i = 0; i < DIM_HEIGHT; ++i) {
            for (int j = 0; j < DIM_WIDTH; ++j) {
                imgData.putFloat((float)mat.get(i,j)[0]);
            }
        }
    }

    @Override
    protected void onPause() {
        super.onPause();
        if (cameraBridgeViewBase!=null){
            cameraBridgeViewBase.disableView();
        }
    }

    @Override
    protected void onResume() {
        super.onResume();

        if (!OpenCVLoader.initDebug()){
            Toast.makeText(this, "There is a problem with open cv", Toast.LENGTH_SHORT).show();
        }
        else{
            baseLoaderCallback.onManagerConnected(BaseLoaderCallback.SUCCESS);
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (cameraBridgeViewBase!=null){
            cameraBridgeViewBase.disableView();
        }
    }
}
