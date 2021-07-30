using System;
using System.Collections;
using System.Collections.Generic;
using RosSharp.RosBridgeClient.MessageTypes.Sensor;
using UnityEngine;
using UnityEngine.UI;
using System.Threading;


namespace RosSharp.RosBridgeClient
{
    [RequireComponent(typeof(RosConnector))]
    public class PointCloudSubscriber : UnitySubscriber<MessageTypes.Sensor.PointCloud2>
    {
        private byte[] byteArray;
        private bool isMessageReceived = false;
        bool readyToProcessMessage = true;
        private int size;

        private Vector3[] pcl;
        private Color[] pcl_color;

        int width;
        int height;
        int row_step;
        int point_step;

        protected override void Start()
        {
            base.Start();

        }

        public void Update()
        {

            if (isMessageReceived)
            {
                PointCloudRendering();
                isMessageReceived = false;
            }


        }

        protected override void ReceiveMessage(PointCloud2 message)
        {


            size = message.data.GetLength(0);

            byteArray = new byte[size];
            byteArray = message.data;


            width = (int)message.width;
            height = (int)message.height;
            row_step = (int)message.row_step;
            point_step = (int)message.point_step;

            size = size / point_step;
            isMessageReceived = true;
        }

        //点群の座標を変換
        void PointCloudRendering()
        {
            pcl = new Vector3[size];
            pcl_color = new Color[size];

            int x_posi;
            int y_posi;
            int z_posi;
            int b_posi;
            int g_posi;
            int r_posi;

            float x;
            float y;
            float z;
            float r;
            float g;
            float b;

            //この部分でbyte型をfloatに変換         
            for (int n = 0; n < size; n++)
            {
                x_posi = n * point_step + 0;
                y_posi = n * point_step + 4;
                z_posi = n * point_step + 8;
                b_posi = n * point_step + 12;
                g_posi = n * point_step + 16;
                r_posi = n * point_step + 20;

                x = BitConverter.ToSingle(byteArray, x_posi);
                y = BitConverter.ToSingle(byteArray, y_posi);
                z = BitConverter.ToSingle(byteArray, z_posi);
                b = BitConverter.ToSingle(byteArray, b_posi);
                g = BitConverter.ToSingle(byteArray, g_posi);
                r = BitConverter.ToSingle(byteArray, r_posi);


                pcl[n] = new Vector3(x, z, y);
                pcl_color[n] = new Color(r, g, b);

            }
        }

        public Vector3[] GetPCL()
        {
            return pcl;
        }

        public Color[] GetPCLColor()
        {
            return pcl_color;
        }
    }
}
