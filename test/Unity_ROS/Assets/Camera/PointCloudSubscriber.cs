using System;
using System.Collections;
using System.Collections.Generic;
using UnityEditor;
using UnityEngine;
using UnityEngine.UI;
using System.Threading;
using Unity.Robotics.ROSTCPConnector;
using RosPointCloud = RosMessageTypes.Sensor.MPointCloud2;
using RosPointField = RosMessageTypes.Sensor.MPointField;
using Intel.RealSense;

public class PointCloudSubscriber : RsFrameProvider
{
    public ImageSubscriber colorImageSubscriber;
    public ImageSubscriber depthImageSubscriber;

    /// <summary>
    /// The parallelism mode of the module
    /// </summary>
    public enum ProcessMode
    {
        Multithread,
        UnityThread,
    }

    // public static RsDevice Instance { get; private set; }

    /// <summary>
    /// Threading mode of operation, Multithread or UnityThread
    /// </summary>
    [Tooltip("Threading mode of operation, Multithreads or Unitythread")]
    public ProcessMode processMode;

    // public bool Streaming { get; private set; }

    /// <summary>
    /// Notifies upon streaming start
    /// </summary>
    public override event Action<PipelineProfile> OnStart;

    /// <summary>
    /// Notifies when streaming has stopped
    /// </summary>
    public override event Action OnStop;

    /// <summary>
    /// Fired when a new frame is available
    /// </summary>
    public override event Action<Frame> OnNewSample;

    private Thread worker;
    private readonly AutoResetEvent stopEvent = new AutoResetEvent(false);

    void OnEnable()
    {
        if (processMode == ProcessMode.Multithread)
        {
            stopEvent.Reset();
            worker = new Thread(WaitForFrames);
            worker.IsBackground = true;
            worker.Start();
        }

        StartCoroutine(WaitAndStart());
    }

    IEnumerator WaitAndStart()
    {
        yield return new WaitForEndOfFrame();
        Streaming = true;
        if (OnStart != null)
            OnStart(ActiveProfile);
    }

    void OnDisable()
    {
        OnNewSample = null;
        // OnNewSampleSet = null;

        if (worker != null)
        {
            stopEvent.Set();
            worker.Join();
        }

        if (Streaming && OnStop != null)
            OnStop();

        if (ActiveProfile != null)
        {
            ActiveProfile.Dispose();
            ActiveProfile = null;
        }

        Streaming = false;
    }

    void OnDestroy()
    {
        // OnStart = null;
        OnStop = null;

        if (ActiveProfile != null)
        {
            ActiveProfile.Dispose();
            ActiveProfile = null;
        }
    }

    private void RaiseSampleEvent(Frame frame)
    {
        var onNewSample = OnNewSample;
        if (onNewSample != null)
        {
            onNewSample(frame);
        }
    }

    /// <summary>
    /// Worker Thread for multithreaded operations
    /// </summary>
    private void WaitForFrames()
    {
        while (!stopEvent.WaitOne(0))
        {
            //Debug.Log("PCS: WaitForFrames");
            //FrameSet frames;
            //using (var frames = m_pipeline.WaitForFrames())
            //using(frames)
            //    RaiseSampleEvent(frames);
        }
    }

    void Update()
    {
        if (!Streaming)
            return;

        if (processMode != ProcessMode.UnityThread)
            return;

        //Debug.Log("PCS: Update");

        //FrameSet frames;
        //        public bool PollForFrames(out FrameSet result)
        // {
        //     object error;
        //     IntPtr fs;
        //     if (NativeMethods.rs2_pipeline_poll_for_frames(Handle, out fs, out error) > 0)
        //     {
        //         result = FrameSet.Create(fs);
        //         return true;
        //     }

        //     result = null;
        //     return false;
        // }

        //if (m_pipeline.PollForFrames(out frames))
        //{
        //    using (frames)
        //        RaiseSampleEvent(frames);
        //}
    }
}