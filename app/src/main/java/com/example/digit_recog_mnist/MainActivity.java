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
        mat2 = new Mat(width,height, CvType.CV_8UC4);
        mat3 = new Mat(width,height, CvType.CV_8UC4);
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
        mat1 = inputFrame.rgba();
        Core.transpose(mat1,mat2);
        Core.flip(mat2,mat1,1);

        int w = mat1.width();
        int h = mat1.height();

        Imgproc.rectangle(mat1, new Point(w * 1 / 3, h * 1 / 3), new Point(
                w * 2 / 3, h * 2 /  3 ), new Scalar( 255, 0, 0 ), 5
        );
        Rect roi  = new Rect(w*1/3,h*1/3,(w*2/3 - w*1/3),(h*2/3-h*1/3));
        Mat temp = mat1.submat(roi);


        Imgproc.resize(temp,imageToanalyse,imageToanalyse.size(),0,0,0);

        img = Bitmap.createBitmap(imageToanalyse.cols(), imageToanalyse.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(imageToanalyse, img);

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
