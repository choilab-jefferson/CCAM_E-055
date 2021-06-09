#if UNITY_STANDALONE
#define IMPORT_GLENABLE
#endif

using UnityEngine;
using System;
using System.Collections;
using System.Runtime.InteropServices;

public class enableOpenGL : MonoBehaviour
{
    const UInt32 GL_VERTEX_PROGRAM_POINT_SIZE = 0x8642;
    const UInt32 GL_POINT_SMOOTH = 0x0B10;

    const string LibGLPath = "/usr/lib/libGL.so";

#if IMPORT_GLENABLE
    [DllImport(LibGLPath)]
    public static extern void glEnable(UInt32 cap);

    private bool mIsOpenGL;

    void Start()
    {
        mIsOpenGL = SystemInfo.graphicsDeviceVersion.Contains("OpenGL");
    }

    void OnPreRender()
    {
        if (mIsOpenGL)
        {
            glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);
            glEnable(GL_POINT_SMOOTH);
        }

    }
#endif
}