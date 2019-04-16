// Upgrade NOTE: replaced '_Object2World' with 'unity_ObjectToWorld'
// Upgrade NOTE: replaced 'mul(UNITY_MATRIX_MVP,*)' with 'UnityObjectToClipPos(*)'

// History:
// 
// 18-02-12:
//     Fix for new wireframe edge code
// 
// 18-02-05:
//     Made transtion center model space instead of world
//     Added toggle to hide X and Y planes
// 
// 18-12-13:  global input option for blob positions

Shader "Bounding_Box" {

Properties {

    [Header(Wireframe)]
        _Near_Width_("Near Width", Range(0,1)) = 0.01
        _Far_Width_("Far Width", Range(0,1)) = 0.05
        _Near_Distance_("Near Distance", Range(0,10)) = 1
        _Far_Distance_("Far Distance", Range(0,20)) = 5
        _Edge_Color_("Edge Color", Color) = (0.27451,0.27451,0.27451,1)
     
    [Header(Proximity)]
        _Proximity_Max_Intensity_("Proximity Max Intensity", Range(0,1)) = 0.59
        _Proximity_Far_Radius_("Proximity Far Radius", Range(0,1)) = 0.51
        _Proximity_Near_Radius_("Proximity Near Radius", Range(0,1)) = 0.01
     
    [Header(Blob)]
        [Toggle] _Blob_Enable_("Blob Enable", Float) = 1
        _Blob_Position_("Blob Position", Vector) = (0.7, 0, 0, 1)
        _Blob_Intensity_("Blob Intensity", Range(0,3)) = 0.6
        _Blob_Near_Size_("Blob Near Size", Range(0,1)) = 0.03
        _Blob_Far_Size_("Blob Far Size", Range(0,1)) = 0.06
        _Blob_Near_Distance_("Blob Near Distance", Range(0,1)) = 0
        _Blob_Far_Distance_("Blob Far Distance", Range(0,1)) = 0.08
        _Blob_Fade_Length_("Blob Fade Length", Range(0,1)) = 0.08
        _Blob_Inner_Fade_("Blob Inner Fade", Range(0.01,1)) = 0.1
        _Blob_Pulse_("Blob Pulse", Range(0,1)) = 0
        _Blob_Fade_("Blob Fade", Range(0,1)) = 1
     
    [Header(Blob Texture)]
        [NoScaleOffset] _Blob_Texture_("Blob Texture", 2D) = "" {}
     
    [Header(Blob 2)]
        [Toggle] _Blob_Enable_2_("Blob Enable 2", Float) = 0
        _Blob_Position_2_("Blob Position 2", Vector) = (-1, 0, 0, 1)
        _Blob_Near_Size_2_("Blob Near Size 2", Float) = 0.02
        _Blob_Inner_Fade_2_("Blob Inner Fade 2", Range(0,1)) = 0.1
        _Blob_Pulse_2_("Blob Pulse 2", Range(0,1)) = 0
        _Blob_Fade_2_("Blob Fade 2", Range(0,1)) = 1
     
    [Header(Transition)]
        [Toggle(_ENABLE_TRANSITION_)] _Enable_Transition_("Enable Transition", Float) = 0
        _Center_("Center", Vector) = (0.5, 0.3, 0.2, 1)
        _Transition_("Transition", Range(0,1)) = 0.92
        _Radius_("Radius", Range(0,5)) = 4
        _Fuzz_("Fuzz", Range(0,1)) = 0.92
        _Start_Time_("Start Time", Float) = 0
        _Transition_Period_("Transition Period", Range(0,5)) = 1
        _Flash_Color_("Flash Color", Color) = (0.635294,0,1,1)
        _Trim_Color_("Trim Color", Color) = (0.113725,0,1,1)
        [Toggle] _Invert_("Invert", Float) = 1
     
    [Header(Hololens Edge Fade)]
        [Toggle(_ENABLE_FADE_)] _Enable_Fade_("Enable Fade", Float) = 1
        _Fade_Width_("Fade Width", Range(0,10)) = 1.5
     
    [Header(Hide Faces)]
        [Toggle] _Hide_XY_Faces_("Hide XY Faces", Float) = 0
     
    [Header(Debug)]
        [Toggle] _Show_Frame_("Show Frame", Float) = 1
     

    [Header(Global)]
        [Toggle] Use_Global_Left_Index("Use Global Left Index", Float) = 0
        [Toggle] Use_Global_Right_Index("Use Global Right Index", Float) = 0
}

SubShader {
    Tags { "RenderType" = "Transparent" "Queue" = "Transparent" }
    Blend One One
    Cull Off
    ZWrite Off
    Tags {"DisableBatching" = "True"}

    LOD 100


    Pass

    {

    CGPROGRAM

    #pragma vertex vert
    #pragma fragment frag
    #pragma multi_compile_instancing
    #pragma target 4.0
    #pragma multi_compile _ _ENABLE_FADE_
    #pragma multi_compile _ _ENABLE_TRANSITION_

    #include "UnityCG.cginc"

    sampler2D _Blob_Texture_;
    //bool _Enable_Fade_;
    float _Fade_Width_;
    bool _Blob_Enable_2_;
    float3 _Blob_Position_2_;
    float _Blob_Near_Size_2_;
    float _Blob_Inner_Fade_2_;
    float _Blob_Pulse_2_;
    float _Blob_Fade_2_;
    bool _Blob_Enable_;
    float3 _Blob_Position_;
    float _Blob_Intensity_;
    float _Blob_Near_Size_;
    float _Blob_Far_Size_;
    float _Blob_Near_Distance_;
    float _Blob_Far_Distance_;
    float _Blob_Fade_Length_;
    float _Blob_Inner_Fade_;
    float _Blob_Pulse_;
    float _Blob_Fade_;
    float _Proximity_Max_Intensity_;
    float _Proximity_Far_Radius_;
    float _Proximity_Near_Radius_;
    bool _Hide_XY_Faces_;
    //bool _Enable_Transition_;
    float3 _Center_;
    float _Transition_;
    float _Radius_;
    float _Fuzz_;
    float _Start_Time_;
    float _Transition_Period_;
    float4 _Flash_Color_;
    float4 _Trim_Color_;
    bool _Invert_;
    bool _Show_Frame_;
    float _Near_Width_;
    float _Far_Width_;
    float _Near_Distance_;
    float _Far_Distance_;
    float4 _Edge_Color_;

    bool Use_Global_Left_Index;
    bool Use_Global_Right_Index;
    float4 Global_Left_Index_Tip_Position;
    float4 Global_Right_Index_Tip_Position;
    float4 Global_Left_Thumb_Tip_Position;
    float4 Global_Right_Thumb_Tip_Position;



    struct VertexInput {
        float4 vertex : POSITION;
        half3 normal : NORMAL;
        float2 uv0 : TEXCOORD0;
        float4 tangent : TANGENT;
        float4 color : COLOR;
        UNITY_VERTEX_INPUT_INSTANCE_ID
    };

    struct VertexOutput {
        float4 pos : SV_POSITION;
        half4 normalWorld : TEXCOORD5;
        float2 uv : TEXCOORD0;
        float3 posWorld : TEXCOORD7;
        float4 tangent : TANGENT;
        float4 binormal : TEXCOORD6;
        float4 extra1 : TEXCOORD4;
      UNITY_VERTEX_OUTPUT_STEREO
    };

    // declare parm vars here

    //BLOCK_BEGIN Blob_Vertex 78

    void Blob_Vertex_B78(
        float3 Position,
        float3 Normal,
        float3 Tangent,
        float3 Bitangent,
        float3 Blob_Position,
        float Intensity,
        float4 Near_Color,
        float4 Far_Color,
        float Blob_Near_Size,
        float Blob_Far_Size,
        float Blob_Near_Distance,
        float Blob_Far_Distance,
        float4 Vx_Color,
        float2 UV,
        float3 Face_Center,
        float2 Face_Size,
        float2 In_UV,
        float Blob_Fade_Length,
        float Inner_Fade,
        float Blob_Enabled,
        float Fade,
        float Pulse,
        float Visible,
        out float3 Out_Position,
        out float2 Out_UV,
        out float3 Blob_Info    )
    {
        
        float Hit_Distance = dot(Blob_Position-Face_Center, Normal);
        float3 Hit_Position = Blob_Position - Hit_Distance * Normal;
        
        float absD = abs(Hit_Distance);
        float lerpVal = clamp((absD-Blob_Near_Distance)/(Blob_Far_Distance-Blob_Near_Distance),0.0,1.0);
        float fadeIn = 1.0-clamp((absD-Blob_Far_Distance)/Blob_Fade_Length,0.0,1.0);
        
        //compute blob position & uv
        float3 delta = Hit_Position - Face_Center;
        float2 blobCenterXY = float2(dot(delta,Tangent),dot(delta,Bitangent));
        
        float innerFade = 1.0-clamp(-Hit_Distance/Inner_Fade,0.0,1.0);
        
        float size = lerp(Blob_Near_Size,Blob_Far_Size,lerpVal)*innerFade*Blob_Enabled*Visible;
        //float size = lerp(Blob_Near_Size,sqrt(max(0.0,radius*radius-Hit_Distance*Hit_Distance)),lerpVal);
        
        float2 quadUVin = 2.0*UV-1.0;  // remap to (-.5,.5)
        float2 blobXY = blobCenterXY+quadUVin*size;
        //keep the quad within the face
        float2 blobClipped = clamp(blobXY,-Face_Size*0.5,Face_Size*0.5);
        float2 blobUV = (blobClipped-blobCenterXY)/max(size,0.0001)*2.0;
        
        float3 blobCorner = Face_Center + blobClipped.x*Tangent + blobClipped.y*Bitangent;
        
        //blend using VxColor.r=1 for blob quad, 0 otherwise
        Out_Position = lerp(Position,blobCorner,Vx_Color.rrr);
        Out_UV = lerp(In_UV,blobUV,Vx_Color.rr);
        Blob_Info = float3((lerpVal*0.5+0.5)*(1.0-Pulse),Intensity*fadeIn*Fade,0.0);
        
    }
    //BLOCK_END Blob_Vertex

    //BLOCK_BEGIN Object_To_World_Pos 75

    void Object_To_World_Pos_B75(
        float3 Pos_Object,
        out float3 Pos_World    )
    {
        Pos_World=(mul(unity_ObjectToWorld, float4(Pos_Object, 1)));
        
    }
    //BLOCK_END Object_To_World_Pos

    //BLOCK_BEGIN Holo_Edge_Vertex 80

    void Holo_Edge_Vertex_B80(
        float3 Normal,
        float2 UV,
        float3 Tangent,
        float3 Bitangent,
        float3 Incident,
        bool Hide_Faces,
        out float4 Holo_Edges    )
    {
        float NdotI = dot(Incident,Normal);
        float2 flip = (UV-float2(0.5,0.5));
        
        float udot = dot(Incident,Tangent)*flip.x*NdotI;
        float uval = (udot>0.0 && !Hide_Faces ? 0.0 : 1.0);
        
        float vdot = -dot(Incident,Bitangent)*flip.y*NdotI;
        float vval = (vdot>0.0 && !Hide_Faces ? 0.0 : 1.0);
        
        float frontside = NdotI<0.0 || Hide_Faces ? 1.0 : 0.0;
        //float smoothall = Hide_Faces ? 0.0 : 1.0;
        Holo_Edges = float4(1.0,1.0,1.0,1.0)-float4(uval*UV.x,uval*(1.0-UV.x),vval*UV.y,vval*(1.0-UV.y)) * frontside;
    }
    //BLOCK_END Holo_Edge_Vertex

    //BLOCK_BEGIN Choose_Blob 73

    void Choose_Blob_B73(
        float4 Vx_Color,
        float3 Position1,
        float3 Position2,
        bool Blob_Enable_1,
        bool Blob_Enable_2,
        float Near_Size_1,
        float Near_Size_2,
        float Blob_Inner_Fade_1,
        float Blob_Inner_Fade_2,
        float Blob_Pulse_1,
        float Blob_Pulse_2,
        float Blob_Fade_1,
        float Blob_Fade_2,
        out float3 Position,
        out float Near_Size,
        out float Inner_Fade,
        out float Blob_Enable,
        out float Fade,
        out float Pulse    )
    {
        float3 blob1 =  (Use_Global_Left_Index ? Global_Left_Index_Tip_Position.xyz :  Position1);
        float3 blob2 =  (Use_Global_Right_Index ? Global_Right_Index_Tip_Position.xyz :  Position2);
        
        Position = blob1*(1.0-Vx_Color.g)+Vx_Color.g*blob2;
        
        float b1 = Blob_Enable_1 ? 1.0 : 0.0;
        float b2 = Blob_Enable_2 ? 1.0 : 0.0;
        Blob_Enable = b1+(b2-b1)*Vx_Color.g;
        
        Pulse = Blob_Pulse_1*(1.0-Vx_Color.g)+Vx_Color.g*Blob_Pulse_2;
        Fade = Blob_Fade_1*(1.0-Vx_Color.g)+Vx_Color.g*Blob_Fade_2;
        Near_Size = Near_Size_1*(1.0-Vx_Color.g)+Vx_Color.g*Near_Size_2;
        Inner_Fade = Blob_Inner_Fade_1*(1.0-Vx_Color.g)+Vx_Color.g*Blob_Inner_Fade_2;
    }
    //BLOCK_END Choose_Blob

    //BLOCK_BEGIN Wireframe_Vertex 64

    void Wireframe_Vertex_B64(
        float3 Position,
        float3 Normal,
        float3 Tangent,
        float3 Bitangent,
        float3 Tangent_World,
        float3 Bitangent_World,
        float Edge_Width,
        out float3 Result,
        out float2 UV,
        out float2 Widths,
        out float2 Face_Size    )
    {
        Face_Size = float2(length(Tangent_World),length(Bitangent_World));
        Widths.xy = Edge_Width/Face_Size;
        
        float x = dot(Position,Tangent);
        float y = dot(Position,Bitangent);
        
        float dx = 0.5-abs(x);
        float newx = (0.5 - dx * Widths.x * 2.0)*sign(x);
        
        float dy = 0.5-abs(y);
        float newy = (0.5 - dy * Widths.y * 2.0)*sign(y);
        
        Result = Normal * 0.5 + newx * Tangent + newy * Bitangent;
        
        UV.x = dot(Result,Tangent) + 0.5;
        UV.y = dot(Result,Bitangent) + 0.5;
    }
    //BLOCK_END Wireframe_Vertex

    //BLOCK_BEGIN Object_To_World_Dir 66

    void Object_To_World_Dir_B66(
        float3 Dir_Object,
        out float3 Dir_World    )
    {
        Dir_World=(mul((float3x3)unity_ObjectToWorld, Dir_Object));
    }
    //BLOCK_END Object_To_World_Dir

    //BLOCK_BEGIN Object_To_World_Normal 65

    void Object_To_World_Normal_B65(
        float3 Nrm_Object,
        out float3 Nrm_World    )
    {
        Nrm_World=UnityObjectToWorldNormal(Nrm_Object);
    }
    //BLOCK_END Object_To_World_Normal

    //BLOCK_BEGIN ComputeWidth 96

    void ComputeWidth_B96(
        float3 Eye,
        float3 Model_Center,
        float Near_Width,
        float Far_Width,
        float Near_Distance,
        float Far_Distance,
        out float Width    )
    {
        float d = distance(Model_Center, Eye);
        float k = saturate((d-Near_Distance)/(Far_Distance-Near_Distance));
        Width = lerp(Near_Width, Far_Width, k);
        
    }
    //BLOCK_END ComputeWidth


    VertexOutput vert(VertexInput vertInput)
    {
        UNITY_SETUP_INSTANCE_ID(vertInput);
        VertexOutput o;
        UNITY_INITIALIZE_VERTEX_OUTPUT_STEREO(o);


        float3 Pos_World_Q75;
        Object_To_World_Pos_B75(_Center_,Pos_World_Q75);

        float3 Position_Q73;
        float Near_Size_Q73;
        float Inner_Fade_Q73;
        float Blob_Enable_Q73;
        float Fade_Q73;
        float Pulse_Q73;
        Choose_Blob_B73(vertInput.color,_Blob_Position_,_Blob_Position_2_,_Blob_Enable_,_Blob_Enable_2_,_Blob_Near_Size_,_Blob_Near_Size_2_,_Blob_Inner_Fade_,_Blob_Inner_Fade_2_,_Blob_Pulse_,_Blob_Pulse_2_,_Blob_Fade_,_Blob_Fade_2_,Position_Q73,Near_Size_Q73,Inner_Fade_Q73,Blob_Enable_Q73,Fade_Q73,Pulse_Q73);

        // Hide_Faces
        float Visible_Q77 = _Hide_XY_Faces_ ? abs(vertInput.normal.z) : 1.0;

        float3 Dir_World_Q66;
        Object_To_World_Dir_B66(vertInput.tangent,Dir_World_Q66);

        float3 Dir_World_Q67;
        Object_To_World_Dir_B66((normalize(cross(vertInput.normal,vertInput.tangent))),Dir_World_Q67);

        // To_RGBA
        float R_Q47;
        float G_Q47;
        float B_Q47;
        float A_Q47;
        R_Q47=vertInput.color.r; G_Q47=vertInput.color.g; B_Q47=vertInput.color.b; A_Q47=vertInput.color.a;

        float3 Nrm_World_Q65;
        Object_To_World_Normal_B65(vertInput.normal,Nrm_World_Q65);

        // Scale3
        float3 Result_Q52 = 0.5 * vertInput.normal;

        float3 Pos_World_Q95;
        Object_To_World_Pos_B75(float3(0,0,0),Pos_World_Q95);

        // Normalize3
        float3 Normalized_Q50 = normalize(Nrm_World_Q65);

        // Normalize3
        float3 Normalized_Q48 = normalize(Dir_World_Q66);

        // Normalize3
        float3 Normalized_Q49 = normalize(Dir_World_Q67);

        float3 Pos_World_Q51;
        Object_To_World_Pos_B75(Result_Q52,Pos_World_Q51);

        float Width_Q96;
        ComputeWidth_B96(_WorldSpaceCameraPos,Pos_World_Q95,_Near_Width_,_Far_Width_,_Near_Distance_,_Far_Distance_,Width_Q96);

        float3 Result_Q64;
        float2 UV_Q64;
        float2 Widths_Q64;
        float2 Face_Size_Q64;
        Wireframe_Vertex_B64(vertInput.vertex.xyz,vertInput.normal,vertInput.tangent,(normalize(cross(vertInput.normal,vertInput.tangent))),Dir_World_Q66,Dir_World_Q67,Width_Q96,Result_Q64,UV_Q64,Widths_Q64,Face_Size_Q64);

        // Scale3
        float3 Result_Q76 = Visible_Q77 * Result_Q64;

        // Pack_For_Vertex
        float3 Vec3_Q53 = float3(Widths_Q64.x,Widths_Q64.y,R_Q47);

        float3 Pos_World_Q42;
        Object_To_World_Pos_B75(Result_Q76,Pos_World_Q42);

        // Incident3
        float3 Incident_Q63 = normalize(Pos_World_Q42-_WorldSpaceCameraPos);

        float3 Out_Position_Q78;
        float2 Out_UV_Q78;
        float3 Blob_Info_Q78;
        Blob_Vertex_B78(Pos_World_Q42,Normalized_Q50,Normalized_Q48,Normalized_Q49,Position_Q73,_Blob_Intensity_,float4(0.41,0,0.216,1),float4(0,0.089,1,1),Near_Size_Q73,_Blob_Far_Size_,_Blob_Near_Distance_,_Blob_Far_Distance_,vertInput.color,vertInput.uv0,Pos_World_Q51,Face_Size_Q64,UV_Q64,_Blob_Fade_Length_,Inner_Fade_Q73,Blob_Enable_Q73,Fade_Q73,Pulse_Q73,Visible_Q77,Out_Position_Q78,Out_UV_Q78,Blob_Info_Q78);

        float4 Holo_Edges_Q80;
        Holo_Edge_Vertex_B80(Normalized_Q50,vertInput.uv0,Dir_World_Q66,Dir_World_Q67,Incident_Q63,_Hide_XY_Faces_,Holo_Edges_Q80);

        float3 Position = Out_Position_Q78;
        float2 UV = Out_UV_Q78;
        float3 Tangent = Pos_World_Q75;
        float3 Binormal = Blob_Info_Q78;
        float4 Color = float4(1,1,1,1);
        float4 Extra1 = Holo_Edges_Q80;
        float3 Normal = Vec3_Q53;


        o.pos = UnityObjectToClipPos(vertInput.vertex);
        o.pos = mul(UNITY_MATRIX_VP, float4(Position,1));
        o.posWorld = Position;
        o.normalWorld.xyz = Normal; o.normalWorld.w=1.0;
        o.uv = UV;
        o.tangent.xyz = Tangent; o.tangent.w=1.0;
        o.binormal.xyz = Binormal; o.binormal.w=1.0;
        o.extra1=Extra1;

        return o;
    }

    //BLOCK_BEGIN Holo_Edge_Fragment 68

    void Holo_Edge_Fragment_B68(
        float Edge_Width,
        float4 Edges,
        out float NotEdge    )
    {
        float2 c = float2(min(Edges.x,Edges.y),min(Edges.z,Edges.w));
        float2 df = fwidth(c)*Edge_Width;
        float2 g = saturate(c/df);
        NotEdge = g.x*g.y;
    }
    //BLOCK_END Holo_Edge_Fragment

    //BLOCK_BEGIN Blob_Fragment 71

    void Blob_Fragment_B71(
        sampler2D Blob_Texture,
        float2 UV,
        float3 Blob_Info,
        out float4 Blob_Color    )
    {
        float k = dot(UV,UV);
        Blob_Color = Blob_Info.y * tex2D(Blob_Texture,float2(float2(sqrt(k),Blob_Info.x).x,1.0-float2(sqrt(k),Blob_Info.x).y))*(1.0-saturate(k));
    }
    //BLOCK_END Blob_Fragment

    //BLOCK_BEGIN Transition 84

    float tramp(float start, float end, float x)
    {
        return smoothstep(start,end,x);
    //    return saturate((x-start)/(end-start));
    }
    
    void Transition_B84(
        float3 Position,
        float Time,
        float3 Center,
        float Transition,
        float Radius,
        float Fuzz,
        float Start_Time,
        float Speed,
        float4 Flash_Color,
        float4 Trim_Color,
        bool Invert,
        float Edge_Weight,
        out float Trans_Intensity,
        out float4 Flash    )
    {
        float t = Invert ? 1.0-Transition : Transition;
        t = Start_Time>0.0 ? clamp((Time-Start_Time)/Speed,0.0,1.0) : t;
        
        float d = distance(Center,Position);
        float k = t * Radius;
        float s1 = tramp(k-Fuzz-Fuzz,k-Fuzz,d);
        float s2 = tramp(k-Fuzz,k,d);
        
        float s = saturate(s1-s2);
        Trans_Intensity = Invert ? s1 : 1.0-s1;
        //Trans_Intensity = 1; //sqrt(Trans_Intensity);
        
        Flash = Edge_Weight*s*lerp(Trim_Color,Flash_Color,float4(s,s,s,s));
    }
    //BLOCK_END Transition

    //BLOCK_BEGIN Wireframe_Fragment 85

    float2 FilterStep(float2 Edge, float2 X)
    {
        // note we are in effect doubling the filter width
        float2 dX = max(fwidth(X),float2(0.00001,0.00001));
        return saturate( (X+dX - max(Edge,X-dX))/(dX*2.0));
    }
    
    void Wireframe_Fragment_B85(
        float3 Widths,
        float2 UV,
        out float Edge    )
    {
        float2 c = min(UV,float2(1.0,1.0)-UV);
        float2 g = FilterStep(Widths.xy,c); 
        Edge = 1.0-min(g.x,g.y);
        
    }
    //BLOCK_END Wireframe_Fragment

    //BLOCK_BEGIN Proximity 74

    void Proximity_B74(
        float3 Position,
        float3 Proximity_Center,
        float3 Proximity_Center_2,
        float Proximity_Max_Intensity,
        float Proximity_Far_Radius,
        float Proximity_Near_Radius,
        out float Proximity    )
    {
        float3 blob1 =  (Use_Global_Left_Index ? Global_Left_Index_Tip_Position.xyz :  Proximity_Center);
        float3 blob2 =  (Use_Global_Right_Index ? Global_Right_Index_Tip_Position.xyz :  Proximity_Center_2);
        
        float3 delta1 = blob1-Position;
        float3 delta2 = blob2-Position;
        
        float d2 = sqrt(min(dot(delta1,delta1),dot(delta2,delta2)));
        Proximity = Proximity_Max_Intensity * (1.0-saturate((d2-Proximity_Near_Radius)/(Proximity_Far_Radius-Proximity_Near_Radius)));
    }
    //BLOCK_END Proximity


    //fixed4 frag(VertexOutput fragInput, fixed facing : VFACE) : SV_Target
    half4 frag(VertexOutput fragInput) : SV_Target
    {
        half4 result;

        float NotEdge_Q68;
        #if defined(_ENABLE_FADE_)
          Holo_Edge_Fragment_B68(_Fade_Width_,fragInput.extra1,NotEdge_Q68);
        #else
          NotEdge_Q68 = 1;
        #endif

        float4 Blob_Color_Q71;
        Blob_Fragment_B71(_Blob_Texture_,fragInput.uv,fragInput.binormal.xyz,Blob_Color_Q71);

        // To_XYZ
        float X_Q62;
        float Y_Q62;
        float Z_Q62;
        X_Q62=fragInput.normalWorld.xyz.x;
        Y_Q62=fragInput.normalWorld.xyz.y;
        Z_Q62=fragInput.normalWorld.xyz.z;

        float Edge_Q85;
        Wireframe_Fragment_B85(fragInput.normalWorld.xyz,fragInput.uv,Edge_Q85);

        float Proximity_Q74;
        Proximity_B74(fragInput.posWorld,_Blob_Position_,_Blob_Position_2_,_Proximity_Max_Intensity_,_Proximity_Far_Radius_,_Proximity_Near_Radius_,Proximity_Q74);

        float Trans_Intensity_Q84;
        float4 Flash_Q84;
        #if defined(_ENABLE_TRANSITION_)
          Transition_B84(fragInput.posWorld,_Time.y,fragInput.tangent.xyz,_Transition_,_Radius_,_Fuzz_,_Start_Time_,_Transition_Period_,_Flash_Color_,_Trim_Color_,_Invert_,Edge_Q85,Trans_Intensity_Q84,Flash_Q84);
        #else
          Trans_Intensity_Q84 = 0;
          Flash_Q84 = float4(0,0,0,0);
        #endif

        // Max
        float MaxAB_Q59=max(Proximity_Q74,Trans_Intensity_Q84);

        // Multiply
        float Edge_Intensity_Q70 = Edge_Q85 * MaxAB_Q59;

        // Add_Scaled_Color
        float4 Wire_Color_Q61 = Flash_Q84 + Edge_Intensity_Q70 * _Edge_Color_;

        // Mix_Colors
        float4 Color_At_T_Q60 = lerp(Wire_Color_Q61, Blob_Color_Q71,float4( Z_Q62, Z_Q62, Z_Q62, Z_Q62));

        // Conditional_Color
        float4 Result_Q69 = _Show_Frame_ ? float4(0.3,0.3,0.3,0.3) : Color_At_T_Q60;

        // Scale_Color
        float4 Result_Q46 = NotEdge_Q68 * Result_Q69;

        float4 Out_Color = Result_Q46;
        float Clip_Threshold = 0;
        bool To_sRGB = false;

        result = Out_Color;
        return result;
    }

    ENDCG
  }
 }
}
