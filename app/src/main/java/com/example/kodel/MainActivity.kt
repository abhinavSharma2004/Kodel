package com.example.kodel

import android.graphics.Bitmap
import android.graphics.ImageDecoder
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.provider.MediaStore
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import com.example.kodel.ml.MobilenetV110224Quant
import com.example.kodel.ui.theme.KodelTheme
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.common.FileUtil.loadLabels
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.label.TensorLabel
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.nio.ByteOrder

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            KodelTheme {
                // A surface container using the 'background' color from the theme
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background
                ) {
                    getImage()
                }
            }
        }
    }
}

@Composable
private fun classify(bitmap: Bitmap): String{
        val context = LocalContext.current
        val model = MobilenetV110224Quant.newInstance(context)
        val resize = Bitmap.createScaledBitmap(bitmap, 224, 224, true)
        val inputImage = TensorImage(DataType.UINT8)
        inputImage.load(resize)
        val inputBuffer = TensorBuffer.createFixedSize(intArrayOf(1, 224, 224, 3), DataType.UINT8)

        val byteBuffer = inputImage.buffer
        byteBuffer.order(ByteOrder.nativeOrder())
        inputBuffer.loadBuffer(byteBuffer)

        val outputs = model.process(inputBuffer)

        val outputBuffer = outputs.outputFeature0AsTensorBuffer
        val labels = loadLabels(context, "labels.txt")
        val labeledProbability = TensorLabel(labels, outputBuffer).mapWithFloatValue
        val maxValue = labeledProbability.values.maxOrNull()
        val res = labeledProbability.filterValues { it == maxValue }.keys.first()

        model.close()

        return res
    }

@Composable
fun getImage() {
    var imageUri by remember { mutableStateOf<Uri?>(null) }
    val context = LocalContext.current
    val bitmap =  remember { mutableStateOf<Bitmap?>(null) }
    val launcher = rememberLauncherForActivityResult(contract =
    ActivityResultContracts.GetContent()) { uri: Uri? ->
        imageUri = uri
    }
    var res by remember {
        mutableStateOf("")
    }

    Column(
        modifier = Modifier.fillMaxSize(),
        verticalArrangement = Arrangement.Center,
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Button(onClick = {
            launcher.launch("image/*")},
            modifier = Modifier
                .fillMaxWidth()
                .padding(4.dp)
                .padding(start = 24.dp, end = 24.dp)) {
            Text(text = "Select image from device")
        }

        Spacer(modifier = Modifier.height(12.dp))

        imageUri?.let {
            if (Build.VERSION.SDK_INT < 28) {
                bitmap.value = MediaStore.Images
                    .Media.getBitmap(context.contentResolver,it)

            } else {
                val source = ImageDecoder
                    .createSource(context.contentResolver,it)
                bitmap.value = ImageDecoder.decodeBitmap(source)
            }

            bitmap.value?.let {  btm ->
                Image(bitmap = btm.asImageBitmap(),
                    contentDescription = "image to process",
                    modifier = Modifier.size(400.dp))
            }
        }

//        Button(onClick = {
//            res = bitmap.value?.let { classify(bitmap = it) }.toString()
//            },
//            modifier = Modifier
//                .fillMaxWidth()
//                .padding(4.dp)
//                .padding(start = 24.dp, end = 24.dp)) {
//            Text(text = "Classify Image")
//        }
//
//        Spacer(modifier = Modifier.height(12.dp))

        res = bitmap.value?.let { classify(bitmap = it) }.toString()

        Text(text = "The image shows "+res)

    }
}



@Preview(showBackground = true)
@Composable
fun GreetingPreview() {
    KodelTheme {
        getImage()
    }
}