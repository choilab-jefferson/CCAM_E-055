using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosCompressedImage = RosMessageTypes.Sensor.MCompressedImage;
//using RosImage = RosMessageTypes.Sensor.MImage;

public class ImageSubscriber : MonoBehaviour
{
    // ROS Connector
    private ROSConnection ros;

    public string topicName;

    public MeshRenderer meshRenderer;

    private Texture2D texture2D;
    private byte[] imageData;
    private enum PixelFormat
    {
        RGB8,
        _16UC1,
    }
    private enum ImageFormat
    {
        Jpeg,
        PNG,
    }
    private PixelFormat pixelFormat;
    private ImageFormat imageFormat;
    private bool isMessageReceived;


    void Start()
    {
        texture2D = new Texture2D(1, 1, TextureFormat.R16, true);
        meshRenderer.material = new Material(Shader.Find("Standard"));

        ros = ROSConnection.instance;
        ros.Subscribe<RosCompressedImage>(topicName, ReceiveMessage);
        Debug.Log("Image: " + topicName);
    }

    void ReceiveMessage(RosCompressedImage imageMsg)
    {
        string[] formats = imageMsg.format.Split(new char[] {';'});
        if(formats[0] == "rgb8")
        {
            pixelFormat = PixelFormat.RGB8;
        }
        else
        {
            pixelFormat = PixelFormat._16UC1;
        }

        string format = "";
        if(formats.Length > 1)
            format = formats[1].Split(new char[] {' '})[1];
        if(format == "jpeg")
        {
            imageFormat = ImageFormat.Jpeg;
        }
        else if(format == "png")
        {
            imageFormat = ImageFormat.PNG;
        }
        
        imageData = imageMsg.data;
        isMessageReceived = true;
        Debug.Log("Camera: " + topicName + " " + formats[0] + " " + format);
        //Debug.Log("Camera: " + topicName + " " + imageMsg.encoding);
    }

    private void Update()
    {
        if (isMessageReceived)
            ProcessMessage();
    }

    private void ProcessMessage()
    {
        texture2D.LoadImage(imageData);
        texture2D.Apply();
        meshRenderer.material.SetTexture("_MainTex", texture2D);
        isMessageReceived = false;
    }
}