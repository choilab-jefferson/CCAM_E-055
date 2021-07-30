using UnityEngine;
using System.Collections;
using System.Text;

[ExecuteInEditMode ()]
public class AllocMem : MonoBehaviour {
	
	public bool show = true;
	public bool showFPS = false;
	public bool showInEditor = false;
	public void Start () {
		useGUILayout = false;
	}
	
	// Use this for initialization
	public void OnGUI () {
		if (!show || (!Application.isPlaying && !showInEditor)) {
			return;
		}
		
		int collCount = System.GC.CollectionCount (0);
		
		if (lastCollectNum != collCount) {
			lastCollectNum = collCount;
			delta = Time.realtimeSinceStartup-lastCollect;
			lastCollect = Time.realtimeSinceStartup;
			lastDeltaTime = Time.deltaTime;
			collectAlloc = allocMem;
		}
		
		allocMem = (int)System.GC.GetTotalMemory (false);
		
		peakAlloc = allocMem > peakAlloc ? allocMem : peakAlloc;
		
		if (Time.realtimeSinceStartup - lastAllocSet > 0.3F) {
			int diff = allocMem - lastAllocMemory;
			lastAllocMemory = allocMem;
			lastAllocSet = Time.realtimeSinceStartup;
			
			if (diff >= 0) {
				allocRate = diff;
			}
		}
		
		StringBuilder text = new StringBuilder ();
		
		text.Append ("Currently allocated			");
		text.Append ((allocMem/1000000F).ToString ("0"));
		text.Append ("mb\n");
		
		text.Append ("Peak allocated				");
		text.Append ((peakAlloc/1000000F).ToString ("0"));
		text.Append ("mb (last	collect ");
		text.Append ((collectAlloc/1000000F).ToString ("0"));
		text.Append (" mb)\n");
		
		
		text.Append ("Allocation rate				");
		text.Append ((allocRate/1000000F).ToString ("0.0"));
		text.Append ("mb\n");
		
		text.Append ("Collection frequency		");
		text.Append (delta.ToString ("0.00"));
		text.Append ("s\n");
		
		text.Append ("Last collect delta			");
		text.Append (lastDeltaTime.ToString ("0.000"));
		text.Append ("s (");
		text.Append ((1F/lastDeltaTime).ToString ("0.0"));
		
		text.Append (" fps)");
		
		if (showFPS) {
			text.Append ("\n"+(1F/Time.deltaTime).ToString ("0.0")+" fps");
		}
		
		GUI.Box (new Rect (5,5,310,80+(showFPS ? 16 : 0)),"");
		GUI.Label (new Rect (10,5,1000,200),text.ToString ());
		/*GUI.Label (new Rect (5,5,1000,200),
			"Currently allocated			"+(allocMem/1000000F).ToString ("0")+"mb\n"+
			"Peak allocated				"+(peakAlloc/1000000F).ToString ("0")+"mb "+
			("(last	collect"+(collectAlloc/1000000F).ToString ("0")+" mb)" : "")+"\n"+
			"Allocation rate				"+(allocRate/1000000F).ToString ("0.0")+"mb\n"+
			"Collection space			"+delta.ToString ("0.00")+"s\n"+
			"Last collect delta			"+lastDeltaTime.ToString ("0.000") + " ("+(1F/lastDeltaTime).ToString ("0.0")+")");*/
	}
	
	private float lastCollect = 0;
	private float lastCollectNum = 0;
	private float delta = 0;
	private float lastDeltaTime = 0;
	private int allocRate = 0;
	private int lastAllocMemory = 0;
	private float lastAllocSet = -9999;
	private int allocMem = 0;
	private int collectAlloc = 0;
	private int peakAlloc = 0;
	
}