"use strict";(self.webpackChunksynapseml=self.webpackChunksynapseml||[]).push([[5122],{3905:function(e,n,t){t.d(n,{Zo:function(){return p},kt:function(){return f}});var r=t(67294);function a(e,n,t){return n in e?Object.defineProperty(e,n,{value:t,enumerable:!0,configurable:!0,writable:!0}):e[n]=t,e}function l(e,n){var t=Object.keys(e);if(Object.getOwnPropertySymbols){var r=Object.getOwnPropertySymbols(e);n&&(r=r.filter((function(n){return Object.getOwnPropertyDescriptor(e,n).enumerable}))),t.push.apply(t,r)}return t}function o(e){for(var n=1;n<arguments.length;n++){var t=null!=arguments[n]?arguments[n]:{};n%2?l(Object(t),!0).forEach((function(n){a(e,n,t[n])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(t)):l(Object(t)).forEach((function(n){Object.defineProperty(e,n,Object.getOwnPropertyDescriptor(t,n))}))}return e}function i(e,n){if(null==e)return{};var t,r,a=function(e,n){if(null==e)return{};var t,r,a={},l=Object.keys(e);for(r=0;r<l.length;r++)t=l[r],n.indexOf(t)>=0||(a[t]=e[t]);return a}(e,n);if(Object.getOwnPropertySymbols){var l=Object.getOwnPropertySymbols(e);for(r=0;r<l.length;r++)t=l[r],n.indexOf(t)>=0||Object.prototype.propertyIsEnumerable.call(e,t)&&(a[t]=e[t])}return a}var s=r.createContext({}),c=function(e){var n=r.useContext(s),t=n;return e&&(t="function"==typeof e?e(n):o(o({},n),e)),t},p=function(e){var n=c(e.components);return r.createElement(s.Provider,{value:n},e.children)},u={inlineCode:"code",wrapper:function(e){var n=e.children;return r.createElement(r.Fragment,{},n)}},m=r.forwardRef((function(e,n){var t=e.components,a=e.mdxType,l=e.originalType,s=e.parentName,p=i(e,["components","mdxType","originalType","parentName"]),m=c(t),f=a,k=m["".concat(s,".").concat(f)]||m[f]||u[f]||l;return t?r.createElement(k,o(o({ref:n},p),{},{components:t})):r.createElement(k,o({ref:n},p))}));function f(e,n){var t=arguments,a=n&&n.mdxType;if("string"==typeof e||a){var l=t.length,o=new Array(l);o[0]=m;var i={};for(var s in n)hasOwnProperty.call(n,s)&&(i[s]=n[s]);i.originalType=e,i.mdxType="string"==typeof e?e:a,o[1]=i;for(var c=2;c<l;c++)o[c]=t[c];return r.createElement.apply(null,o)}return r.createElement.apply(null,t)}m.displayName="MDXCreateElement"},5721:function(e,n,t){t.r(n),t.d(n,{assets:function(){return p},contentTitle:function(){return s},default:function(){return f},frontMatter:function(){return i},metadata:function(){return c},toc:function(){return u}});var r=t(83117),a=t(80102),l=(t(67294),t(3905)),o=["components"],i={title:"Mlflow Installation",description:"install Mlflow on different environments"},s=void 0,c={unversionedId:"mlflow/installation",id:"mlflow/installation",title:"Mlflow Installation",description:"install Mlflow on different environments",source:"@site/docs/mlflow/installation.md",sourceDirName:"mlflow",slug:"/mlflow/installation",permalink:"/SynapseML/docs/next/mlflow/installation",draft:!1,tags:[],version:"current",frontMatter:{title:"Mlflow Installation",description:"install Mlflow on different environments"},sidebar:"docs",previous:{title:"Introduction",permalink:"/SynapseML/docs/next/mlflow/introduction"},next:{title:"Examples",permalink:"/SynapseML/docs/next/mlflow/examples"}},p={},u=[{value:"Installation",id:"installation",level:2},{value:"Install Mlflow on Databricks",id:"install-mlflow-on-databricks",level:3},{value:"Install Mlflow on Synapse",id:"install-mlflow-on-synapse",level:3},{value:"Create Azure Machine Learning Workspace",id:"create-azure-machine-learning-workspace",level:4},{value:"Create an Azure ML Linked Service",id:"create-an-azure-ml-linked-service",level:4},{value:"Auth Synapse Workspace",id:"auth-synapse-workspace",level:4},{value:"Use Mlflow in Synapse",id:"use-mlflow-in-synapse",level:4}],m={toc:u};function f(e){var n=e.components,t=(0,a.Z)(e,o);return(0,l.kt)("wrapper",(0,r.Z)({},m,t,{components:n,mdxType:"MDXLayout"}),(0,l.kt)("h2",{id:"installation"},"Installation"),(0,l.kt)("p",null,"Install MLflow from PyPI via ",(0,l.kt)("inlineCode",{parentName:"p"},"pip install mlflow")),(0,l.kt)("p",null,"MLflow requires ",(0,l.kt)("inlineCode",{parentName:"p"},"conda")," to be on the ",(0,l.kt)("inlineCode",{parentName:"p"},"PATH")," for the projects feature."),(0,l.kt)("p",null,"Learn more about MLflow on their ",(0,l.kt)("a",{parentName:"p",href:"https://github.com/mlflow/mlflow"},"GitHub page"),"."),(0,l.kt)("h3",{id:"install-mlflow-on-databricks"},"Install Mlflow on Databricks"),(0,l.kt)("p",null,"If you're using Databricks, install Mlflow with this command:"),(0,l.kt)("pre",null,(0,l.kt)("code",{parentName:"pre"},"# run this so that Mlflow is installed on workers besides driver\n%pip install mlflow\n")),(0,l.kt)("h3",{id:"install-mlflow-on-synapse"},"Install Mlflow on Synapse"),(0,l.kt)("p",null,"Mlflow is pre-installed on Synapse. To log model with Mlflow, you need to create an Azure Machine Learning workspace and link it with your Synapse workspace."),(0,l.kt)("h4",{id:"create-azure-machine-learning-workspace"},"Create Azure Machine Learning Workspace"),(0,l.kt)("p",null,"Follow this document to create ",(0,l.kt)("a",{parentName:"p",href:"https://learn.microsoft.com/en-us/azure/machine-learning/quickstart-create-resources#create-the-workspace"},"AML workspace"),", compute instance and compute clusters aren't required."),(0,l.kt)("h4",{id:"create-an-azure-ml-linked-service"},"Create an Azure ML Linked Service"),(0,l.kt)("img",{src:"https://mmlspark.blob.core.windows.net/graphics/Documentation/ml_linked_service_1.png",width:"600"}),(0,l.kt)("ul",null,(0,l.kt)("li",{parentName:"ul"},"In the Synapse workspace, go to ",(0,l.kt)("strong",{parentName:"li"},"Manage")," -> ",(0,l.kt)("strong",{parentName:"li"},"External connections")," -> ",(0,l.kt)("strong",{parentName:"li"},"Linked services"),", select ",(0,l.kt)("strong",{parentName:"li"},"+ New")),(0,l.kt)("li",{parentName:"ul"},"Select the workspace you want to log the model in and create the linked service. You need the ",(0,l.kt)("strong",{parentName:"li"},"name of the linked service")," to set up connection.")),(0,l.kt)("h4",{id:"auth-synapse-workspace"},"Auth Synapse Workspace"),(0,l.kt)("img",{src:"https://mmlspark.blob.core.windows.net/graphics/Documentation/ml_linked_service_2.png",width:"600"}),(0,l.kt)("ul",null,(0,l.kt)("li",{parentName:"ul"},"Go to the ",(0,l.kt)("strong",{parentName:"li"},"Azure Machine Learning workspace")," resource -> ",(0,l.kt)("strong",{parentName:"li"},"access control (IAM)")," -> ",(0,l.kt)("strong",{parentName:"li"},"Role assignment"),", select ",(0,l.kt)("strong",{parentName:"li"},"+ Add"),", choose ",(0,l.kt)("strong",{parentName:"li"},"Add role assignment")),(0,l.kt)("li",{parentName:"ul"},"Choose ",(0,l.kt)("strong",{parentName:"li"},"contributor"),", select next"),(0,l.kt)("li",{parentName:"ul"},"In members page, choose ",(0,l.kt)("strong",{parentName:"li"},"Managed identity"),", select  ",(0,l.kt)("strong",{parentName:"li"},"+ select members"),". Under ",(0,l.kt)("strong",{parentName:"li"},"managed identity"),", choose Synapse workspace. Under ",(0,l.kt)("strong",{parentName:"li"},"Select"),", choose the workspace you run your experiment on. Click ",(0,l.kt)("strong",{parentName:"li"},"Select"),", ",(0,l.kt)("strong",{parentName:"li"},"Review + assign"),".")),(0,l.kt)("h4",{id:"use-mlflow-in-synapse"},"Use Mlflow in Synapse"),(0,l.kt)("p",null,"Set up connection"),(0,l.kt)("pre",null,(0,l.kt)("code",{parentName:"pre",className:"language-python"},'\n#AML\xa0workspace\xa0authentication\xa0using\xa0linked\xa0service\nfrom\xa0notebookutils.mssparkutils\xa0import\xa0azureML\nlinked_service_name = "YourLinkedServiceName"\nws\xa0=\xa0azureML.getWorkspace(linked_service_name)\nmlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())\n\n#Set\xa0MLflow\xa0experiment.\xa0\nexperiment_name\xa0=\xa0"synapse-mlflow-experiment"\nmlflow.set_experiment(experiment_name)\xa0\n')))}f.isMDXComponent=!0}}]);