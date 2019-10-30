package com.example.digit_recog_mnist;

import androidx.appcompat.app.AppCompatActivity;

import android.graphics.Bitmap;
import android.os.Bundle;
import android.util.Log;
import android.view.Surface;
import android.view.SurfaceView;
import android.widget.ImageView;
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

public class MainActivity extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2 {

    CameraBridgeViewBase cameraBridgeViewBase;
    BaseLoaderCallback baseLoaderCallback;
    Mat mat1,mat2,mat3;
    Mat imageToanalyse;
    ImageView myImageView;
    Bitmap img;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        cameraBridgeViewBase = (JavaCameraView)findViewById(R.id.myCameraView);
        cameraBridgeViewBase.setVisibility(SurfaceView.VISIBLE);
        cameraBridgeViewBase.setCvCameraViewListener(this);
        //OpenCVLoader.initDebug()
        myImageView = findViewById(R.id.ImageView);


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

        ///use this to classify camera feed

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
