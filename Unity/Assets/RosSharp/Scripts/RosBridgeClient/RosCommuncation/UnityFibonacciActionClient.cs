/*
© Siemens AG, 2019
Author: Berkay Alp Cakal (berkay_alp.cakal.ct@siemens.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
<http://www.apache.org/licenses/LICENSE-2.0>.
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

using UnityEngine;
using UnityEngine.UI;

namespace RosSharp.RosBridgeClient.Actionlib
{
    [RequireComponent(typeof(RosConnector))]
    public class UnityFibonacciActionClient : MonoBehaviour
    {
        private RosConnector rosConnector;
        public FibonacciActionClient fibonacciActionClient;

        public string actionName;
        public int fibonacciOrder = 20;
        public string status = "";
        public string feedback = "";
        public string result = "";

        public Text statusText;
        public Text feedbackText;
        public Text resultText;
        private void Start()
        {
            rosConnector = GetComponent<RosConnector>();
            fibonacciActionClient = new FibonacciActionClient(actionName, rosConnector.RosSocket);
            fibonacciActionClient.Initialize();
        }

        private void Update()
        {
            status   = fibonacciActionClient.GetStatusString();
            feedback = fibonacciActionClient.GetFeedbackString();
            result   = fibonacciActionClient.GetResultString();

            statusText.text = "Status: " + fibonacciActionClient.GetStatusString();
            feedbackText.text = "Feedback: " + fibonacciActionClient.GetFeedbackString();
            resultText.text = "Result: " + fibonacciActionClient.GetResultString();
        }

        public void RegisterGoal()
        {
            fibonacciActionClient.fibonacciOrder = fibonacciOrder;
        }

    }
}
