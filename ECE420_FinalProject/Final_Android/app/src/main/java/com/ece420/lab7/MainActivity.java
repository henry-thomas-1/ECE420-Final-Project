package com.ece420.lab7;

import android.Manifest;
import android.content.pm.ActivityInfo;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.support.v4.app.ActivityCompat;
import android.support.v4.content.ContextCompat;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;
import android.view.View;
import android.view.WindowManager;
import android.view.inputmethod.InputMethodManager;
import android.widget.EditText;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.Spinner;
import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Rect2d;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.tracking.TrackerKCF;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.core.MatOfRect;
import org.opencv.core.Rect;
import android.content.Context;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Collections;
import android.widget.Button;


public class MainActivity extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2 {

    private static final String TAG = "MainActivity";

    // variables to determine what mode for the camera we are in:
    int mode_num = 0;
     // 0 = smile detection, 1 = eye detection, 2 = frown detection, 3 = wave detection, 4 = thumbs up, 5 = smart photographer
    private boolean picture = false;
    private boolean ready = false;
    private Mat pic;
    private Button specificButton;
    private boolean isSpecificValueSet = false; // Variable to control visibility

    private Spinner modeSpinner;
    private String selectedMode;
    private void handleModeChange(String mode) {
        if (mode.equals("Smile")) {
            mode_num = 0;
            isSpecificValueSet = false;
            specificButton.setVisibility(View.INVISIBLE); // Make the button invisible
            editText2.setVisibility(View.INVISIBLE);
        } else if (mode.equals("Eye")) {
            mode_num = 1;
            isSpecificValueSet = false;
            specificButton.setVisibility(View.INVISIBLE); // Make the button invisible
            editText2.setVisibility(View.VISIBLE);
        } else if (mode.equals("Frown")){
            mode_num = 2;
            isSpecificValueSet = false;
            specificButton.setVisibility(View.INVISIBLE); // Make the button invisible
            editText2.setVisibility(View.INVISIBLE);
        } else if (mode.equals("Wave")){
            mode_num = 3;
            isSpecificValueSet = false;
            specificButton.setVisibility(View.INVISIBLE); // Make the button invisible
            editText2.setVisibility(View.INVISIBLE);
        }else if (mode.equals("Thumbs")){
            mode_num = 4;
            isSpecificValueSet = false;
            specificButton.setVisibility(View.INVISIBLE); // Make the button invisible
            editText2.setVisibility(View.INVISIBLE);
        }else if (mode.equals("Smart Cam")){
            mode_num = 5;
            isSpecificValueSet = true;
            specificButton.setVisibility(View.VISIBLE);
            specificButton.setText("Ready");
            editText2.setVisibility(View.INVISIBLE);
        }
    }
    private void performSpecificTask() {
        // Code to execute when the button is clicked
        if (picture){
            Log.d(TAG, "performSpecificTask: setting pic and ready to false");
            picture = false;
            ready = false;
            specificButton.setText("Ready");
        }else if(!ready){
            Log.d(TAG, "performSpecificTask: setting ready to true");
            ready = true;
            specificButton.setText("Reset");

        }
    }
    // Declare OpenCV based camera view base
    private CameraBridgeViewBase mOpenCvCameraView;
    private int myWidth;
    private int myHeight;
    private Mat mRgba;
    private Mat mGray;
    //cascade classifiers:
    public static CascadeClassifier loadCascadeClassifier(Context context, int resourceId) {
        CascadeClassifier cascadeClassifier = new CascadeClassifier();
        try {
            // Copy the XML file from resources to app's internal storage
            InputStream inputStream = context.getResources().openRawResource(resourceId);
            File cascadeFile = File.createTempFile("haarcascade", ".xml", context.getCacheDir());
            FileOutputStream outputStream = new FileOutputStream(cascadeFile);
            byte[] buffer = new byte[4096];
            int bytesRead;
            while ((bytesRead = inputStream.read(buffer)) != -1) {
                outputStream.write(buffer, 0, bytesRead);
            }
            inputStream.close();
            outputStream.close();

            // Load the cascade classifier from the copied XML file
            cascadeClassifier.load(cascadeFile.getAbsolutePath());

            // Delete the temporary file after loading the classifier
            cascadeFile.delete();

        } catch (IOException e) {
            e.printStackTrace();
        }
        return cascadeClassifier;
    }
    public static Mat readCSVToMat(InputStream inputStream, int rows, int cols) throws IOException {
        Mat matrix = new Mat(rows, cols, CvType.CV_32FC1);

        BufferedReader reader = new BufferedReader(new InputStreamReader(inputStream));
        String line;
        List<String[]> data = new ArrayList<>();
        int rowIndex = 0;

        while ((line = reader.readLine()) != null) {
            String[] parts = line.split(",");
            data.add(parts);
            for (int colIndex = 0; colIndex < parts.length; colIndex++) {
                matrix.put(rowIndex, colIndex, Float.parseFloat(parts[colIndex]));
            }
            rowIndex++;
        }

        reader.close();

        return matrix;
    }
    Mat w1_eyes;
    Mat w2_eyes;
    Mat b1_eyes;
    Mat w1_smile;
    Mat w2_smile;
    Mat b1_smile;
    Mat w1_frown;
    Mat b1_frown;
    Mat w2_frown;
    double b2_frown = -0.00025319;
    double b2_smile = 0.23020589;
    double b2_eyes = -0.02759429;
    CascadeClassifier faceCascade;
    CascadeClassifier mouthCascade;
    CascadeClassifier eyeCascade;
    public double sigmoid(double x) {
        return 1 / (1 + Math.exp(-x));
    }
    public Mat sigmoid(Mat x) {
        Mat result = x.clone();
        for(int i = 0; i < x.rows(); i++){
            for(int j = 0; j < x.cols(); j++){
                double[] val = x.get(i, j);
                result.put(i, j, sigmoid(val[0]));
            }
        }
        return result;
    }
    public int classify2layer(Mat W1, Mat b1, Mat W2, double b2, Mat image) {
        // Check dimensions compatibility
        image.convertTo(image, CvType.CV_64F);
        W1.convertTo(W1, CvType.CV_64F);
        W2.convertTo(W2, CvType.CV_64F);
        b1.convertTo(b1, CvType.CV_64F);

        // Calculate dot product using gemm
        Mat hiddenOutput = new Mat();
        Core.gemm(image, W1, 1, new Mat(), 1, hiddenOutput);
        Core.add(hiddenOutput, b1, hiddenOutput);
        String what = "Hidden Ouput = {";
        for(int i = 0; i < W1.cols(); i++){
            what += Arrays.toString(hiddenOutput.get(0,i));
            what += ", ";
        }
        Log.d(TAG, "classify2layer: "+what);
        hiddenOutput = sigmoid(hiddenOutput);
        String what2 = "Hidden Ouput = {";
        for(int i = 0; i < W1.cols(); i++){
            what2 += Arrays.toString(hiddenOutput.get(0,i));
            what2 += ", ";
        }
        Log.d(TAG, "classify2layer: "+what2);
        Mat dotProductResult = new Mat();
        double finalOutput = 0;
        Core.gemm(hiddenOutput, W2, 1, new Mat(), 1, dotProductResult);
        double dotProductValue = dotProductResult.get(0, 0)[0];
        finalOutput = sigmoid(dotProductValue + b2);
        Log.d(TAG, "classify2layer: finalOutput for classifier = "+finalOutput);

        // Classify based on final output
        if (finalOutput <= 0.5) {
            return 0;
        } else {
            return 1;
        }
    }
    // Method for dot product
    //moving average calculation
    ArrayList<Integer> queue = new ArrayList<>(Arrays.asList(0,0,0,0,0,0,0,0,0,0));
    ArrayList<Integer> queue2 = new ArrayList<>(Arrays.asList(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0));
    ArrayList<Integer> queue3 = new ArrayList<>(Arrays.asList(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0));

    double moving_ave(ArrayList<Integer> q, int value){
        q.remove(0);
        q.add(value);
        int sum = 0;
        for(int num : q){
            sum += num;
        }
        return ((double)sum) / ((double)q.size());
    }
    //layout items:
    EditText editText;
    EditText editText2;
    InputMethodManager imm;
    CascadeClassifier handCascade;
    CascadeClassifier palmCascade;

    // Declare variables for hand detection
    private long lastHandTime = System.currentTimeMillis();
    private int movementThreshold = 20;
    private int minWavingDistance = 30;
    private boolean isWaving = false;
    private boolean kernelActive = false;
    private int[] kernelBuffer = new int[20];

    private TrackerKCF myTracker;
    private Rect2d myROI = new Rect2d(0,0,0,0);

    // Declare variables for hand detection
    private int bufferIndex = 0;
    private Context context;

    private boolean trackingStarted = false;
    private Mat roiMat;
    private Mat theta;
    private int flag = 1;
    private int number_detected = -1;
    private int mode = 1;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        setContentView(R.layout.activity_main);
        super.setRequestedOrientation (ActivityInfo.SCREEN_ORIENTATION_LANDSCAPE);

        // Request User Permission on Camera
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_DENIED){
            ActivityCompat.requestPermissions(this, new String[] {Manifest.permission.CAMERA}, 1);}

        // OpenCV Loader and Avoid using OpenCV Manager
        if (!OpenCVLoader.initDebug()) {
            Log.e(this.getClass().getSimpleName(), "  OpenCVLoader.initDebug(), not working.");
        } else {
            Log.d(this.getClass().getSimpleName(), "  OpenCVLoader.initDebug(), working.");
        }

        faceCascade = loadCascadeClassifier(MainActivity.this, R.raw.haarcascade_frontalface_default);
        mouthCascade = loadCascadeClassifier(MainActivity.this, R.raw.haarcascade_mouth);
        eyeCascade = loadCascadeClassifier(MainActivity.this, R.raw.haarcascade_eye);

        InputStream inputStream = getResources().openRawResource(R.raw.w1_eye);
        try {
            w1_eyes = readCSVToMat(inputStream, 4096, 8);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        inputStream = getResources().openRawResource(R.raw.w2_eye);
        try {
            w2_eyes = readCSVToMat(inputStream, 8, 1);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        inputStream = getResources().openRawResource(R.raw.b1_eye);
        try {
            b1_eyes = readCSVToMat(inputStream, 1, 8);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        inputStream = getResources().openRawResource(R.raw.w1_smile);
        try {
            w1_smile = readCSVToMat(inputStream, 4096, 16);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        inputStream = getResources().openRawResource(R.raw.w2_smile);
        try {
            w2_smile = readCSVToMat(inputStream, 16, 1);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        inputStream = getResources().openRawResource(R.raw.b1_smile);
        try {
            b1_smile = readCSVToMat(inputStream, 1, 16);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        inputStream = getResources().openRawResource(R.raw.w1_frown);
        try {
            w1_frown = readCSVToMat(inputStream, 4096, 16);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        inputStream = getResources().openRawResource(R.raw.w2_frown);
        try {
            w2_frown = readCSVToMat(inputStream, 16, 1);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        inputStream = getResources().openRawResource(R.raw.b1_frown);
        try {
            b1_frown = readCSVToMat(inputStream, 1, 16);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        theta = new Mat(4, (64 * 64 + 1), CvType.CV_32FC1);
        inputStream = getResources().openRawResource(R.raw.theta);
        try {
            theta = readCSVToMat(inputStream, 4, (64*64+1));
        } catch (IOException e) {
            throw new RuntimeException(e);
        }


        modeSpinner = (Spinner) findViewById(R.id.mode_spinner);
        ArrayAdapter<CharSequence> adapter = ArrayAdapter.createFromResource(this, R.array.mode_array, android.R.layout.simple_spinner_item);
        adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        modeSpinner.setAdapter(adapter);
        modeSpinner.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> parent, View view, int position, long id) {
                selectedMode = parent.getItemAtPosition(position).toString();
                handleModeChange(selectedMode);
            }
            @Override
            public void onNothingSelected(AdapterView<?> parent) {
                // Do nothing
            }
        });

        specificButton = (Button) findViewById(R.id.specific_button);

        // Check if the variable is set to a specific value
        if (isSpecificValueSet) {
            specificButton.setVisibility(View.VISIBLE); // Make the button visible
        } else {
            specificButton.setVisibility(View.INVISIBLE); // Make the button invisible
        }

        specificButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                // Perform specific task when the button is clicked
                performSpecificTask();
            }
        });

        editText = (EditText) findViewById(R.id.editText);
        editText.setText("No Face");
        editText2 = (EditText) findViewById(R.id.editText2);

        handCascade = loadCascadeClassifier(MainActivity.this, R.raw.fist);
        palmCascade = loadCascadeClassifier(MainActivity.this, R.raw.palm);

        // Setup OpenCV Camera View
        mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.opencv_camera_preview);
        // Use main camera with 0 or front camera with 1
        mOpenCvCameraView.setCameraIndex(1);
        // Force camera resolution, ignored since OpenCV automatically select best ones
        mOpenCvCameraView.setMaxFrameSize(1280, 720);
        mOpenCvCameraView.setCvCameraViewListener(this);

    }


    @Override
    protected void onResume() {
        super.onResume();
        if (!OpenCVLoader.initDebug()) {
            Log.d(TAG, "Internal OpenCV library not found. Using OpenCV Manager for initialization");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION, this, mLoaderCallback);
        } else {
            Log.d(TAG, "OpenCV library found inside package. Using it!");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }

    @Override
    protected void onPause() {
        super.onPause();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS: {
                    Log.i(TAG, "OpenCV loaded successfully");
                    mOpenCvCameraView.enableView();
                    myWidth = mOpenCvCameraView.getWidth();
                    myHeight = mOpenCvCameraView.getHeight();
                    Log.d(TAG, "onCreate: camera dimension ("+myWidth+"x"+myHeight+")");
                }
                break;
                default: {
                    super.onManagerConnected(status);
                }
                break;
            }
        }
    };

    // OpenCV Camera Functionality Code
    @Override
    public void onCameraViewStarted(int width, int height) {
        mRgba = new Mat(height, width, CvType.CV_8UC4);
        mGray = new Mat(height, width, CvType.CV_8UC1);
    }

    @Override
    public void onCameraViewStopped() {
        mRgba.release();
        mGray.release();
    }

    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {

        // Timer
        long start = Core.getTickCount();
        // Grab camera frame in rgba and grayscale format
        mRgba = inputFrame.rgba();
        Mat mRgbacopy = mRgba.clone();
        if(picture == false){
            pic = mRgbacopy;
        }
        // Grab camera frame in gray format
        mGray = inputFrame.gray();

        switch (mode_num) {
            case 0:
                // smile detection
                MatOfRect faces_0 = new MatOfRect();
                faceCascade.detectMultiScale(mGray, faces_0, 1.3, 5, 0, new Size(200, 200), new Size(700, 700));
                for (Rect face : faces_0.toArray()) {
                    int x = face.x;
                    int y = face.y;
                    int w = face.width;
                    int h = face.height;
                    Imgproc.rectangle(mRgba, new org.opencv.core.Point(x, y), new org.opencv.core.Point(x + w, y + h), new Scalar(255, 0, 0), 2);
                    Mat faceRoi = mGray.submat(new Rect(x, y, w, h));
                    Mat lowerHalfFace = mGray.submat(new Rect(x + w / 6, y + h * 5 / 8, w - 2 * (w / 6), h - (h * 5 / 8)));
                    int X = x + (w / 6);
                    int Y = y + (h * 5) / 8;
                    int W = w - 2 * (w / 6);
                    int H = (h * 3) / 8;

                    MatOfRect mouths = new MatOfRect();
                    mouthCascade.detectMultiScale(lowerHalfFace, mouths, 1.2, 12, 0, new Size(50, 50), new Size(300, 300));
                    if (mouths.toArray().length > 0) {
                        Rect mouth = mouths.toArray()[0];
                        int mx = mouth.x;
                        int my = mouth.y;
                        int mw = mouth.width;
                        int mh = mouth.height;

                        Imgproc.rectangle(mRgba, new org.opencv.core.Point(X + mx, Y + my), new org.opencv.core.Point(X + mx + mw, Y + my + mh), new Scalar(0, 255, 0), 2);
                        Mat mouthRoi = mGray.submat(new Rect(X + mx, Y + my, mw, mh));
                        Mat equalized = new Mat();
                        Imgproc.resize(mouthRoi, equalized, new Size(64, 64), 0, 0, Imgproc.INTER_CUBIC);
                        Imgproc.equalizeHist(equalized, equalized);
                        equalized.convertTo(equalized, CvType.CV_8U);
                        equalized.reshape(1, equalized.rows() * equalized.cols());
                        byte[] flattenedArray = new byte[(int) (equalized.total())];
                        equalized.get(0, 0, flattenedArray);
                        Mat matOfBytes = new Mat(1, flattenedArray.length, CvType.CV_8UC1);
                        matOfBytes.put(0, 0, flattenedArray);
                        int smile = classify2layer(w1_smile, b1_smile, w2_smile, b2_smile, matOfBytes);
                        double ave;
                        if (smile == 1) {
                            ave = moving_ave(queue, 1);
                        } else {
                            ave = moving_ave(queue, 0);
                        }

                        if (ave <= 0.5) {
                            Imgproc.putText(mRgba, "Smiling", new Point(x, y - 10), Core.FONT_HERSHEY_SIMPLEX, 0.9, new Scalar(0, 255, 0), 2);
                            runOnUiThread(new Runnable() {
                                @Override
                                public void run() {
                                    // Update UI element here
                                    editText.setText("Smiling");
                                }
                            });
                        } else {
                            Imgproc.putText(mRgba, "Not Smiling", new Point(x, y - 10), Core.FONT_HERSHEY_SIMPLEX, 0.9, new Scalar(0, 0, 255), 2);
                            runOnUiThread(new Runnable() {
                                @Override
                                public void run() {
                                    // Update UI element here
                                    editText.setText("Not Smiling");
                                }
                            });

                        }

                    }
                }
                if (faces_0.empty()) {
                    runOnUiThread(new Runnable() {
                        @Override
                        public void run() {
                            // Update UI element here
                            editText.setText("No Face");
                            imm = (InputMethodManager) getSystemService(Context.INPUT_METHOD_SERVICE);
                            imm.hideSoftInputFromWindow(editText.getWindowToken(), 0);
                        }
                    });
                }
                break;
            //-----------------------------------------------------------------------------
            case 1:
                //eye detection
                MatOfRect faces_1 = new MatOfRect();
                faceCascade.detectMultiScale(mGray, faces_1, 1.3, 5, 0, new Size(200, 200), new Size(700, 700));
                for (Rect face : faces_1.toArray()) {
                    int x = face.x;
                    int y = face.y;
                    int w = face.width;
                    int h = face.height;
                    Imgproc.rectangle(mRgba, new org.opencv.core.Point(x, y), new org.opencv.core.Point(x + w, y + h), new Scalar(255, 0, 0), 2);
                    Mat faceRoi = mGray.submat(new Rect(x, y, w, h));
                    Mat upperhalfface = mGray.submat(new Rect(x, y, w, h / 2));
                    MatOfRect eyes = new MatOfRect();
                    eyeCascade.detectMultiScale(upperhalfface, eyes, 1.1, 3, 0, new Size(60, 60), new Size(250, 250));
                    int count = 0;
                    int val = 0;
                    for (Rect eye : eyes.toArray()) {
                        count++;
                        if (count > 2) {
                            break;
                        }
                        int mx = eye.x;
                        int my = eye.y;
                        int mw = eye.width;
                        int mh = eye.height;

                        Imgproc.rectangle(mRgba, new org.opencv.core.Point(x + mx, y + my), new org.opencv.core.Point(x + mx + mw, y + my + mh), new Scalar(0, 255, 0), 2);
                        Mat eyeROI = mGray.submat(new Rect(x + mx, y + my, mw, mh));
                        Mat equalized = new Mat();
                        Imgproc.resize(eyeROI, equalized, new Size(64, 64), 0, 0, Imgproc.INTER_CUBIC);
                        Imgproc.equalizeHist(equalized, equalized);
                        equalized.convertTo(equalized, CvType.CV_8U);
                        equalized.reshape(1, equalized.rows() * equalized.cols());
                        byte[] flattenedArray = new byte[(int) (equalized.total())];
                        equalized.get(0, 0, flattenedArray);
                        Mat matOfBytes = new Mat(1, flattenedArray.length, CvType.CV_8UC1);
                        matOfBytes.put(0, 0, flattenedArray);
                        int eye_det = classify2layer(w1_eyes, b1_eyes, w2_eyes, b2_eyes, matOfBytes);
                        if(eye_det ==1){
                            val += 1;
                        }
                    }
                    double ave = -1;
                    if (count != 0 && val/count <= 0.5) {
                        ave = moving_ave(queue, 0);
                    } else if (count != 0) {
                        ave = moving_ave(queue, 1);
                    }
                    if (ave <= 0.5 && ave >= 0) {
                        Imgproc.putText(mRgba, "Closed", new Point(x, y-10), Core.FONT_HERSHEY_SIMPLEX, .9, new Scalar(0, 0, 255), 2);
                        runOnUiThread(new Runnable() {
                            @Override
                            public void run() {
                                // Update UI element here
                                editText.setText("Closed");
                                editText2.setText("WAKE UP");
                            }
                        });
                    } else if(ave > 0.5){
                        Imgproc.putText(mRgba, "Open", new Point(x, y-10), Core.FONT_HERSHEY_SIMPLEX, .9, new Scalar(0, 255, 0), 2);
                        runOnUiThread(new Runnable() {
                            @Override
                            public void run() {
                                // Update UI element here
                                editText.setText("Open");
                                editText2.setText("");
                            }
                        });
                    }else{
                        Imgproc.putText(mRgba, "No Eye Detected", new Point(x, y-10), Core.FONT_HERSHEY_SIMPLEX, .9, new Scalar(0, 255, 0), 2);
                        runOnUiThread(new Runnable() {
                            @Override
                            public void run() {
                                // Update UI element here
                                editText.setText("No Eye");
                                editText2.setText("");
                            }
                        });
                    }
                }
                break;

            //-----------------------------------------------------------------------------

            case 2:
                //frown detection
                MatOfRect faces_2 = new MatOfRect();
                faceCascade.detectMultiScale(mGray, faces_2, 1.3, 5, 0, new Size(200, 200), new Size(700, 700));
                for (Rect face : faces_2.toArray()) {
                    int x = face.x;
                    int y = face.y;
                    int w = face.width;
                    int h = face.height;
                    Imgproc.rectangle(mRgba, new org.opencv.core.Point(x, y), new org.opencv.core.Point(x + w, y + h), new Scalar(255, 0, 0), 2);
                    Mat faceRoi = mGray.submat(new Rect(x, y, w, h));
                    Mat lowerHalfFace = mGray.submat(new Rect(x + w / 6, y + h * 5 / 8, w - 2 * (w / 6), h - (h * 5 / 8)));
                    int X = x + (w / 6);
                    int Y = y + (h * 5) / 8;
                    int W = w - 2 * (w / 6);
                    int H = (h * 3) / 8;

                    MatOfRect mouths = new MatOfRect();
                    mouthCascade.detectMultiScale(lowerHalfFace, mouths, 1.2, 12, 0, new Size(50, 50), new Size(300, 300));
                    if (mouths.toArray().length > 0) {
                        Rect mouth = mouths.toArray()[0];
                        int mx = mouth.x;
                        int my = mouth.y;
                        int mw = mouth.width;
                        int mh = mouth.height;

                        Imgproc.rectangle(mRgba, new org.opencv.core.Point(X + mx, Y + my), new org.opencv.core.Point(X + mx + mw, Y + my + mh), new Scalar(0, 255, 0), 2);
                        Mat mouthRoi = mGray.submat(new Rect(X + mx, Y + my, mw, mh));
                        Mat equalized = new Mat();
                        Imgproc.resize(mouthRoi, equalized, new Size(64, 64), 0, 0, Imgproc.INTER_CUBIC);
                        Imgproc.equalizeHist(equalized, equalized);
                        equalized.convertTo(equalized, CvType.CV_8U);
                        equalized.reshape(1, equalized.rows() * equalized.cols());
                        byte[] flattenedArray = new byte[(int) (equalized.total())];
                        equalized.get(0, 0, flattenedArray);
                        Mat matOfBytes = new Mat(1, flattenedArray.length, CvType.CV_8UC1);
                        matOfBytes.put(0, 0, flattenedArray);
                        int frown = classify2layer(w1_frown, b1_frown, w2_frown, b2_frown, matOfBytes);
                        double ave;
                        if (frown == 1) {
                            ave = moving_ave(queue, 1);
                        } else {
                            ave = moving_ave(queue, 0);
                        }

                        if (ave <= 0.5) {
                            Imgproc.putText(mRgba, "Frowning", new Point(x, y - 10), Core.FONT_HERSHEY_SIMPLEX, 0.9, new Scalar(0, 255, 0), 2);
                            runOnUiThread(new Runnable() {
                                @Override
                                public void run() {
                                    // Update UI element here
                                    editText.setText("Frowning");
                                }
                            });
                        } else {
                            Imgproc.putText(mRgba, "Not Frowning", new Point(x, y - 10), Core.FONT_HERSHEY_SIMPLEX, 0.9, new Scalar(0, 0, 255), 2);
                            runOnUiThread(new Runnable() {
                                @Override
                                public void run() {
                                    // Update UI element here
                                    editText.setText("Not Frowning");
                                }
                            });

                        }

                    }
                }
                if (faces_2.empty()) {
                    runOnUiThread(new Runnable() {
                        @Override
                        public void run() {
                            // Update UI element here
                            editText.setText("No Face");
                            imm = (InputMethodManager) getSystemService(Context.INPUT_METHOD_SERVICE);
                            imm.hideSoftInputFromWindow(editText.getWindowToken(), 0);
                        }
                    });
                }
                break;

            //-----------------------------------------------------------------------------

            case 3:
                //waving
                if (!kernelActive) {
                    if (myTracker == null) {

                        // Detect hands using the cascade classifier
                        MatOfRect palms = new MatOfRect();
                        MatOfRect hands = new MatOfRect();

                        palmCascade.detectMultiScale(mGray, hands, 1.1, 9, 0, new Size(150, 150), new Size(500, 500));
                        palmCascade.detectMultiScale(mGray, palms, 1.1, 9, 0, new Size(150, 150), new Size(500, 500));
                        // If no hands are detected for a long time, reset tracking and start detecting new hands
                        if (hands.toArray().length == 0 || System.currentTimeMillis() - lastHandTime > 5000) {
                            myTracker = null;
                            isWaving = false;
                        }

                        // If there are detected hands and no tracker is initialized, initialize a new tracker
                        if (hands.toArray().length > 0 && myTracker == null) {
                            Log.d(TAG, "hand detected");
                            // Initialize the KCF tracker with custom parameters
                            myTracker = TrackerKCF.create();
                            Rect rect = hands.toArray()[0];
                            myROI = new Rect2d(rect.x, rect.y, rect.width, rect.height);
                            myTracker.init(mGray, myROI);
                            lastHandTime = System.currentTimeMillis();
                            // Initialize the kernel buffer with the initial x-coordinate of the detected hand
                            kernelBuffer[0] = (int) myROI.x;
                            kernelActive = true;
                        }
                    }
                }

                // If the tracker is initialized and tracking, update it
                if (myTracker != null && kernelActive) {
                    if (myTracker.update(mGray, myROI)) {
                        // Get the horizontal position of the tracked object
                        int center_x = (int) (myROI.x + myROI.width / 2);

                        // Update the circular kernel buffer

                        kernelBuffer[bufferIndex] = center_x;
                        bufferIndex = (bufferIndex+1)%(kernelBuffer.length);

                        // Implement peak-trough detection algorithm using the kernel buffer array
                        List<Integer> peaks = new ArrayList<>();
                        List<Integer> troughs = new ArrayList<>();
                        for (int i = 0; i < kernelBuffer.length; i++) {
                            int prevIndex = (i == 0) ? kernelBuffer.length - 1 : i - 1;
                            int nextIndex = (i == kernelBuffer.length - 1) ? 0 : i + 1;

                            if (kernelBuffer[i] > kernelBuffer[prevIndex] && kernelBuffer[i] > kernelBuffer[nextIndex]) {
                                peaks.add(kernelBuffer[i]);
                            } else if (kernelBuffer[i] < kernelBuffer[prevIndex] && kernelBuffer[i] < kernelBuffer[nextIndex]) {
                                troughs.add(kernelBuffer[i]);
                            }
                        }

                        if (peaks.size() >= 1 && troughs.size() >= 1) {
                            int peak_trough_distance = Collections.min(peaks) - Collections.max(troughs);
                            if (Math.abs(peak_trough_distance) >= minWavingDistance) {
                                isWaving = true;
                            } else {
                                isWaving = false;
                            }
                        }

                        // Draw a bounding box around the tracked object
                        Imgproc.rectangle(mRgba, myROI.tl(), myROI.br(), new Scalar(0, 255, 0), 2);
                    } else {
                        // Tracking failed, reset tracker and kernel active flag
                        myTracker = null;
                        kernelActive = false;
                    }
                }

                // Display the frame with hand detection, tracking, and waving gesture status
                if (isWaving) {
                    Imgproc.putText(mRgba, "is waving", new Point(50, 50), Core.FONT_HERSHEY_SIMPLEX, 1, new Scalar(0, 255, 0), 2);
                } else {
                    Imgproc.putText(mRgba, "not waving", new Point(50, 50), Core.FONT_HERSHEY_SIMPLEX, 1, new Scalar(0, 0, 255), 2);
                }
                break;

            //-----------------------------------------------------------------------------

            case 4:
                //thumbs
                if (myROI != null && myROI.x < mRgba.cols() && myROI.y < mRgba.rows()) {
                    // Resize the frame to 128x256
                    int roiWidth = 128;
                    int roiHeight = 256;
                    int roiX = (mRgba.cols() - roiWidth) / 2;
                    int roiY = (mRgba.rows() - roiHeight) / 2;

                    myROI = new Rect2d(roiX, roiY, roiWidth, roiHeight);
                    Rect ROI = new Rect(roiX, roiY, roiWidth, roiHeight);
                    // Extract the ROI from the grayscale frame
                    Mat roiMat = mGray.submat(ROI);

                    // Scale the ROI to 64x64
                    Mat mResized = new Mat(64, 64, CvType.CV_8UC1);
                    Imgproc.resize(roiMat, mResized, mResized.size(), 0, 0, Imgproc.INTER_AREA);
                    mResized.convertTo(mResized, CvType.CV_32FC1);

                    mResized = mResized.reshape(0, 64 * 64);
                    mResized.convertTo(mResized, CvType.CV_32FC1);

                    double Min = Core.minMaxLoc(mResized).minVal;
                    double Max = Core.minMaxLoc(mResized).maxVal;
                    double[] val1;
                    for (int i = 0; i < mResized.rows(); i++) {
                        val1 = mResized.get(i, 0);
                        val1[0] = Math.floor((val1[0] - Min) * 255 / (Max - Min));
                        mResized.put(i, 0, 255 - val1[0]);
                    }

                    Mat x_val = new Mat(64 * 64 + 1, 1, CvType.CV_32FC1);

                    for (int i = 0; i < x_val.rows(); i++) {
                        if (i == 0)
                            x_val.put(i, 0, 1);
                        else
                            x_val.put(i, 0, mResized.get(i - 1, 0));
                    }

                    Mat yMat = new Mat(4, 1, CvType.CV_32FC1);
                    float[] y_hat = new float[4];

                    // Using Core.gemm, compute yMat = theta * x_val
                    Core.gemm(theta, x_val, 1.0, new Mat(), 0.0, yMat, 0);

                    // Copy yMat to y_hat
                    yMat.get(0, 0, y_hat);

                    // Implement sigmoid function
                    for (int i = 0; i < 4; i++) {
                        y_hat[i] = (float) (1 / (1 + Math.exp(-y_hat[i])));
                    }

                    float maxVal = y_hat[0];
                    for (int i = 1; i < 4; i++) {
                        if (y_hat[i] > maxVal) {
                            maxVal = y_hat[i];
                            number_detected = i;
                        }
                    }
                    // Display the classification result by drawing a rectangle
                    if (number_detected == 1) {
                        Imgproc.rectangle(mRgba, myROI.tl(), myROI.br(), new Scalar(0, 255, 0), 2);
                        Imgproc.putText(mRgba, "Thumbs Down", new Point(myROI.x, myROI.y - 10),
                                Core.FONT_HERSHEY_SIMPLEX, 1.0, new Scalar(0, 255, 0), 2);
                    } else if (number_detected == 0) {
                        Imgproc.rectangle(mRgba, myROI.tl(), myROI.br(), new Scalar(0, 255, 0), 2);
                        Imgproc.putText(mRgba, "Thumbs Up", new Point(myROI.x, myROI.y - 10),
                                Core.FONT_HERSHEY_SIMPLEX, 1.0, new Scalar(0, 255, 0), 2);
                    } else if (number_detected == 2) {
                        Imgproc.rectangle(mRgba, myROI.tl(), myROI.br(), new Scalar(0, 255, 0), 2);
                        Imgproc.putText(mRgba, "Open Hand", new Point(myROI.x, myROI.y - 10),
                                Core.FONT_HERSHEY_SIMPLEX, 1.0, new Scalar(0, 255, 0), 2);
                    } else if (number_detected == 3) {
                        Imgproc.rectangle(mRgba, myROI.tl(), myROI.br(), new Scalar(0, 255, 0), 2);
                        Imgproc.putText(mRgba, "Closed Hand", new Point(myROI.x, myROI.y - 10),
                                Core.FONT_HERSHEY_SIMPLEX, 1.0, new Scalar(0, 255, 0), 2);
                    } else {
                        Imgproc.rectangle(mRgba, myROI.tl(), myROI.br(), new Scalar(0, 255, 0), 2);
                        Imgproc.putText(mRgba, "No Gesture", new Point(myROI.x, myROI.y - 10),
                                Core.FONT_HERSHEY_SIMPLEX, 1.0, new Scalar(0, 255, 0), 2);
                    }
                }
                break;

            //-----------------------------------------------------------------------------
            case 5:
                //smart camera

                if(picture){
                    return pic.clone();
                }else{
                    int eyes_det = 0;
                    int yeses = 0;
                    MatOfRect faces_5 = new MatOfRect();
                    faceCascade.detectMultiScale(mGray, faces_5, 1.3, 5, 0, new Size(200, 200), new Size(700, 700));
                    for (Rect face : faces_5.toArray()) {
                        int x = face.x;
                        int y = face.y;
                        int w = face.width;
                        int h = face.height;
                        Imgproc.rectangle(mRgba, new org.opencv.core.Point(x, y), new org.opencv.core.Point(x + w, y + h), new Scalar(255, 0, 0), 2);
                        Mat faceRoi = mGray.submat(new Rect(x, y, w, h));
                        Mat lowerHalfFace = mGray.submat(new Rect(x + w / 6, y + h * 5 / 8, w - 2 * (w / 6), h - (h * 5 / 8)));
                        int X = x + (w / 6);
                        int Y = y + (h * 5) / 8;
                        int W = w - 2 * (w / 6);
                        int H = (h * 3) / 8;
                        int smile = 0;
                        MatOfRect mouths = new MatOfRect();
                        mouthCascade.detectMultiScale(lowerHalfFace, mouths, 1.2, 12, 0, new Size(50, 50), new Size(300, 300));
                        if (mouths.toArray().length > 0) {
                            Rect mouth = mouths.toArray()[0];
                            int mx = mouth.x;
                            int my = mouth.y;
                            int mw = mouth.width;
                            int mh = mouth.height;

                            Imgproc.rectangle(mRgba, new org.opencv.core.Point(X + mx, Y + my), new org.opencv.core.Point(X + mx + mw, Y + my + mh), new Scalar(0, 255, 0), 2);
                            Mat mouthRoi = mGray.submat(new Rect(X + mx, Y + my, mw, mh));
                            Mat equalized = new Mat();
                            Imgproc.resize(mouthRoi, equalized, new Size(64, 64), 0, 0, Imgproc.INTER_CUBIC);
                            Imgproc.equalizeHist(equalized, equalized);
                            equalized.convertTo(equalized, CvType.CV_8U);
                            equalized.reshape(1, equalized.rows() * equalized.cols());
                            byte[] flattenedArray = new byte[(int) (equalized.total())];
                            equalized.get(0, 0, flattenedArray);
                            Mat matOfBytes = new Mat(1, flattenedArray.length, CvType.CV_8UC1);
                            matOfBytes.put(0, 0, flattenedArray);
                            smile = classify2layer(w1_smile, b1_smile, w2_smile, b2_smile, matOfBytes);
                        }
                        double ave;
                        if (smile == 1) {
                            ave = moving_ave(queue3, 1);
                        } else {
                            ave = moving_ave(queue3, 0);
                        }

                        if (ave <= 0.5) {
                            yeses += 1;
                            Imgproc.putText(mRgba, "Smiling", new Point(x, y - 10), Core.FONT_HERSHEY_SIMPLEX, 0.9, new Scalar(0, 255, 0), 2);
                        } else {
                            Imgproc.putText(mRgba, "Not Smiling", new Point(x, y - 10), Core.FONT_HERSHEY_SIMPLEX, 0.9, new Scalar(0, 0, 255), 2);
                        }
                        Mat upperhalfface = mGray.submat(new Rect(x, y, w, h / 2));
                        MatOfRect eyes = new MatOfRect();
                        eyeCascade.detectMultiScale(upperhalfface, eyes, 1.1, 3, 0, new Size(60, 60), new Size(250, 250));
                        int count = 0;
                        int val = 0;
                        for (Rect eye : eyes.toArray()) {
                            count++;
                            if (count > 2) {
                                break;
                            }
                            int mx = eye.x;
                            int my = eye.y;
                            int mw = eye.width;
                            int mh = eye.height;

                            Imgproc.rectangle(mRgba, new org.opencv.core.Point(x + mx, y + my), new org.opencv.core.Point(x + mx + mw, y + my + mh), new Scalar(0, 255, 0), 2);
                            Mat eyeROI = mGray.submat(new Rect(x + mx, y + my, mw, mh));
                            Mat equalized = new Mat();
                            Imgproc.resize(eyeROI, equalized, new Size(64, 64), 0, 0, Imgproc.INTER_CUBIC);
                            Imgproc.equalizeHist(equalized, equalized);
                            equalized.convertTo(equalized, CvType.CV_8U);
                            equalized.reshape(1, equalized.rows() * equalized.cols());
                            byte[] flattenedArray = new byte[(int) (equalized.total())];
                            equalized.get(0, 0, flattenedArray);
                            Mat matOfBytes = new Mat(1, flattenedArray.length, CvType.CV_8UC1);
                            matOfBytes.put(0, 0, flattenedArray);
                            int eye_det = classify2layer(w1_eyes, b1_eyes, w2_eyes, b2_eyes, matOfBytes);
                            if (eye_det == 1) {
                                val += 1;
                            }
                        }
                        double ave2 = -1;
                        if (count != 0 && val / count <= 0.5) {
                            ave2 = moving_ave(queue2, 0);
                        } else if (count != 0) {
                            ave2 = moving_ave(queue2, 1);
                        }
                        if (ave2 <= 0.5 && ave2 >= 0) {
                            Imgproc.putText(mRgba, "Closed", new Point(x+300, y - 10), Core.FONT_HERSHEY_SIMPLEX, .9, new Scalar(0, 0, 255), 2);
                        } else if (ave2 > 0.5) {
                            yeses += 1;
                            Imgproc.putText(mRgba, "Open", new Point(x+300, y - 10), Core.FONT_HERSHEY_SIMPLEX, .9, new Scalar(0, 255, 0), 2);

                        } else {
                            Imgproc.putText(mRgba, "No Eye Detected", new Point(x+300, y - 10), Core.FONT_HERSHEY_SIMPLEX, .9, new Scalar(0, 255, 0), 2);
                        }
                    }
                        Log.d(TAG, "onCameraFrame: faces: "+faces_5.toArray().length+", yeses: "+yeses);
                    if(ready && ((double) yeses / (double)faces_5.toArray().length) > 1.8){
                        Log.d(TAG, "onCameraFrame: setting picture to true");
                        picture = true;
                    }
                }
                break;
            default:
                //do nothing


        }
        return mRgba;
    }
}