// Upgrade NOTE: replaced '_Object2World' with 'unity_ObjectToWorld'
// Upgrade NOTE: replaced 'mul(UNITY_MATRIX_MVP,*)' with 'UnityObjectToClipPos(*)'

Shader "Thick_Frame" {

Properties {

    [Header(Round Rect)]
        _Line_Width_("Line Width", Range(0,1)) = 0.038
        _Filter_Width_("Filter Width", Range(0,4)) = 1
        _Base_Color_("Base Color", Color) = (0,0,0,1)
        _Line_Color_("Line Color", Color) = (0.53,0.53,0.53,1)
     
    [Header(Blob)]
        [Toggle] _Blob_Enable_("Blob Enable", Float) = 0
        _Blob_Position_("Blob Position", Vector) = (0, 0, 0, 1)
        _Blob_Intensity_("Blob Intensity", Range(0,3)) = 1
        _Blob_Near_Size_("Blob Near Size", Range(0,1)) = 0.02
        _Blob_Far_Size_("Blob Far Size", Range(0,1)) = 0.05
        _Blob_Near_Distance_("Blob Near Distance", Range(0,1)) = 0
        _Blob_Far_Distance_("Blob Far Distance", Range(0,1)) = 0.2
        _Blob_Fade_Length_("Blob Fade Length", Range(0,1)) = 0.2
        _Blob_Pulse_("Blob Pulse", Range(0,1)) = 0
        _Blob_Fade_("Blob Fade", Range(0,1)) = 1
     
    [Header(Blob Texture)]
        [NoScaleOffset] _Blob_Texture_("Blob Texture", 2D) = "" {}
     
    [Header(Blob 2)]
        [Toggle] _Blob_Enable_2_("Blob Enable 2", Float) = 0
        _Blob_Position_2_("Blob Position 2", Vector) = (0, 0, 0.0, 1)
        _Blob_Near_Size_2_("Blob Near Size 2", Range(0,1)) = 0.01
        _Blob_Pulse_2_("Blob Pulse 2", Range(0,1)) = 0
        _Blob_Fade_2_("Blob Fade 2", Range(0,1)) = 1
     
    [Header(Draw Line)]
        _Draw_("Draw", Range(0,1)) = 1
        _Draw_Fuzz_("Draw Fuzz", Range(0.001,1)) = 0.05
        _Draw_Start_("Draw Start", Range(0,1)) = 0
     
    [Header(Line Highlight)]
        _Rate_("Rate", Range(0,1)) = 0.115
        _Highlight_Color_("Highlight Color", Color) = (0.98,0.98,0.98,1)
        _Highlight_Width_("Highlight Width", Range(0,2)) = 0.33
        _Highlight_Transform_("Highlight Transform", Vector) = (1, 1, 0, 0)
     
    [Header(Iridescence)]
        _Iridescence_Intensity_("Iridescence Intensity", Range(0,1)) = 0.4
        [NoScaleOffset] _Iridescence_Ramp_("Iridescence Ramp", 2D) = "" {}
        _Left_X0_("Left X0", Range(0,1)) = 0.0
        _Left_X1_("Left X1", Range(0,1)) = 0.95
        _Right_X0_("Right X0", Range(0,1)) = 0.05
        _Right_X1_("Right X1", Range(0,1)) = 1
        _Angle_("Angle", Range(-45,45)) = -45
        _Iridescence_Center_("Iridescence Center", Vector) = (0, 0, 0, 1)
     
    [Header(Fade)]
        _Fade_Out_("Fade Out", Range(0,1)) = 1
     
    [Header(Antialiasing)]
        [Toggle] _Smooth_Edges_("Smooth Edges", Float) = 1
     

    [Header(Global)]
        [Toggle] Use_Global_Left_Index("Use Global Left Index", Float) = 0
        [Toggle] Use_Global_Right_Index("Use Global Right Index", Float) = 0
}

SubShader {
    Tags{ "RenderType" = "AlphaTest" "Queue" = "AlphaTest"}
    Blend One OneMinusSrcAlpha
    Tags {"DisableBatching" = "True"}

    LOD 100


    Pass

    {

    CGPROGRAM

    #pragma vertex vert
    #pragma fragment frag
    #pragma multi_compile_instancing
    #pragma target 4.0

    #include "UnityCG.cginc"

    sampler2D _Blob_Texture_;
    float _Draw_;
    float _Draw_Fuzz_;
    float _Draw_Start_;
    bool _Blob_Enable_2_;
    float3 _Blob_Position_2_;
    float _Blob_Near_Size_2_;
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
    float _Blob_Pulse_;
    float _Blob_Fade_;
    float _Fade_Out_;
    float _Rate_;
    float4 _Highlight_Color_;
    float _Highlight_Width_;
    float4 _Highlight_Transform_;
    float _Line_Width_;
    float _Filter_Width_;
    float4 _Base_Color_;
    float4 _Line_Color_;
    float _Iridescence_Intensity_;
    sampler2D _Iridescence_Ramp_;
    float _Left_X0_;
    float _Left_X1_;
    float _Right_X0_;
    float _Right_X1_;
    float _Angle_;
    float3 _Iridescence_Center_;
    bool _Smooth_Edges_;

    bool Use_Global_Left_Index;
    bool Use_Global_Right_Index;
    float4 Global_Left_Index_Tip_Position;
    float4 Global_Right_Index_Tip_Position;
    float4 Global_Left_Thumb_Tip_Position;
    float4 Global_Right_Thumb_Tip_Position;



    struct VertexInput {
        float4 vertex : POSITION;
        half3 normal : NORMAL;
        float4 tangent : TANGENT;
        float4 color : COLOR;
        UNITY_VERTEX_INPUT_INSTANCE_ID
    };

    struct VertexOutput {
        float4 pos : SV_POSITION;
        half4 normalWorld : TEXCOORD5;
        float2 uv : TEXCOORD0;
        float4 tangent : TANGENT;
        float4 binormal : TEXCOORD6;
        float4 extra1 : TEXCOORD4;
        float4 extra2 : TEXCOORD3;
        float4 extra3 : TEXCOORD2;
      UNITY_VERTEX_OUTPUT_STEREO
    };

    // declare parm vars here

    //BLOCK_BEGIN Object_To_World_Pos 300

    void Object_To_World_Pos_B300(
        float3 Pos_Object,
        out float3 Pos_World    )
    {
        Pos_World=(mul(unity_ObjectToWorld, float4(Pos_Object, 1)));
        
    }
    //BLOCK_END Object_To_World_Pos

    //BLOCK_BEGIN Iridescence_Vertex 327

    void Iridescence_Vertex_B327(
        float Intensity,
        sampler2D Texture,
        float Left_X0,
        float Left_X1,
        float Right_X0,
        float Right_X1,
        float Angle,
        float2 UV,
        float TdotI,
        out float3 Iridescence    )
    {
        float k = TdotI*0.5+0.5;
        
        float x = lerp(Left_X0,Left_X1,k);
        float4 left = tex2D(Texture,float2(x,0.5),float2(0,0),float2(0,0));
        
        x = lerp(Right_X0,Right_X1,k);
        float4 right = tex2D(Texture,float2(x,0.5),float2(0,0),float2(0,0));
        
        float2 XY = UV - float2(0.5,0.5);
        //float s = saturate(dot(XY,Axis)+0.5);
        float angle = radians(Angle);
        float s = (cos(angle)*XY.x - sin(angle)*XY.y)/cos(angle);
        Iridescence = Intensity*(left.rgb + s*(right.rgb-left.rgb));
        
        
    }
    //BLOCK_END Iridescence_Vertex

    //BLOCK_BEGIN Round_Rect_Vertex 345

    void Round_Rect_Vertex_B345(
        float2 UV,
        float3 Tangent,
        float3 Binormal,
        float Line_Width,
        float Anisotropy,
        float Gradient1,
        float Gradient2,
        out float2 Rect_UV,
        out float4 Rect_Parms,
        out float2 Scale_XY    )
    {
        Scale_XY = float2(Anisotropy,1.0);
        Rect_UV = (UV - float2(0.5,0.5)) * Scale_XY;
        Rect_Parms.xy = Scale_XY*0.5-float2(Line_Width,Line_Width);
        Rect_Parms.z = Gradient1;
        Rect_Parms.w = Gradient2;
    }
    //BLOCK_END Round_Rect_Vertex

    //BLOCK_BEGIN Line_Vertex 340

    void Line_Vertex_B340(
        float2 Scale_XY,
        float Transition,
        float Transition_Fuzz,
        float Transition_Start,
        float2 UV,
        float Time,
        float Rate,
        float4 Highlight_Transform,
        out float3 Line_Vertex_1,
        out float3 Line_Vertex_2    )
    {
        float angle = 3.14160*Transition;
        float sinAngle = sin(angle);
        float cosAngle = cos(angle);
        
        Line_Vertex_1.xy = -Scale_XY*float2(cosAngle,sinAngle);
        Line_Vertex_1.z = -Transition_Fuzz*Transition;
        //Line_Vertex_1.w = 1.0/Transition_Fuzz;
        
        float angle2 = (Transition_Start+Rate*Time) * 2.0 * 3.1416;
        float sinAngle2 = sin(angle2);
        float cosAngle2 = cos(angle2);
        
        float2 xformUV = UV * Highlight_Transform.xy + Highlight_Transform.zw;
        Line_Vertex_2.x = 0.0;
        Line_Vertex_2.y = cosAngle2*xformUV.x-sinAngle2*xformUV.y;
        Line_Vertex_2.z = sinAngle2*xformUV.x+cosAngle2*xformUV.y;
        
    }
    //BLOCK_END Line_Vertex

    //BLOCK_BEGIN Blob_Vertex 329

    void Blob_Vertex_B329(
        float3 Position,
        float3 Normal,
        float3 Tangent,
        float3 Bitangent,
        float3 Blob_Position,
        float Intensity,
        float Blob_Near_Size,
        float Blob_Far_Size,
        float Blob_Near_Distance,
        float Blob_Far_Distance,
        float Blob_Fade_Length,
        float Blob_Pulse,
        float Blob_Fade,
        out float4 Blob_Info    )
    {
        
        float3 blob =  (Use_Global_Left_Index ? Global_Left_Index_Tip_Position.xyz :  Blob_Position);
        float3 delta = blob - Position;
        float dist = dot(Normal,delta);
        
        float lerpValue = saturate((abs(dist)-Blob_Near_Distance)/(Blob_Far_Distance-Blob_Near_Distance));
        float fadeValue = 1.0-clamp((abs(dist)-Blob_Far_Distance)/Blob_Fade_Length,0.0,1.0);
        
        float size = Blob_Near_Size + (Blob_Far_Size-Blob_Near_Size)*lerpValue;
        
        float2 blobXY = float2(dot(delta,Tangent),dot(delta,Bitangent))/(0.0001+size);
        
        float Fade = fadeValue*Intensity*Blob_Fade;
        
        float Distance = (lerpValue*0.5+0.5)*(1.0-Blob_Pulse);
        Blob_Info = float4(blobXY.x,blobXY.y,Distance,Fade);
        
    }
    //BLOCK_END Blob_Vertex

    //BLOCK_BEGIN Blob_Vertex 330

    void Blob_Vertex_B330(
        float3 Position,
        float3 Normal,
        float3 Tangent,
        float3 Bitangent,
        float3 Blob_Position,
        float Intensity,
        float Blob_Near_Size,
        float Blob_Far_Size,
        float Blob_Near_Distance,
        float Blob_Far_Distance,
        float Blob_Fade_Length,
        float Blob_Pulse,
        float Blob_Fade,
        out float4 Blob_Info    )
    {
        
        float3 blob =  (Use_Global_Right_Index ? Global_Right_Index_Tip_Position.xyz :  Blob_Position);
        float3 delta = blob - Position;
        float dist = dot(Normal,delta);
        
        float lerpValue = saturate((abs(dist)-Blob_Near_Distance)/(Blob_Far_Distance-Blob_Near_Distance));
        float fadeValue = 1.0-clamp((abs(dist)-Blob_Far_Distance)/Blob_Fade_Length,0.0,1.0);
        
        float size = Blob_Near_Size + (Blob_Far_Size-Blob_Near_Size)*lerpValue;
        
        float2 blobXY = float2(dot(delta,Tangent),dot(delta,Bitangent))/(0.0001+size);
        
        float Fade = fadeValue*Intensity*Blob_Fade;
        
        float Distance = (lerpValue*0.5+0.5)*(1.0-Blob_Pulse);
        Blob_Info = float4(blobXY.x,blobXY.y,Distance,Fade);
        
    }
    //BLOCK_END Blob_Vertex

    //BLOCK_BEGIN Move_Verts 325

    void Move_Verts_B325(
        float ScaleXY,
        float3 P,
        float Radius,
        out float3 New_P,
        out float2 New_UV,
        out float Radial_Gradient,
        out float3 Radial_Dir    )
    {
        float2 XY = P.xy+float2(0.5,0.5); //UV;
        float2 center = float2(clamp(XY.x,0.25,0.75),clamp(XY.y,0.25,0.75));
        float2 delta = XY - center;
        float2 r2 = 2.0*(Radius)*float2(1.0/ScaleXY,1.0);
        center = (center - float2(0.5,0.5))* (1.0-r2)*2.0;
        float2 xy = delta*r2*2.0+center;
        New_P = float3(xy.x,xy.y,P.z);
        New_UV = float2(xy.x+0.5,xy.y+0.5);
        
        Radial_Gradient = 1.0-length(delta)*4.0;
        Radial_Dir = float3(xy.x-center.x,xy.y-center.y,0.0);
    }
    //BLOCK_END Move_Verts

    //BLOCK_BEGIN Object_To_World_Dir 304

    void Object_To_World_Dir_B304(
        float3 Dir_Object,
        out float3 Binormal_World    )
    {
        Binormal_World = (mul((float3x3)unity_ObjectToWorld, Dir_Object));
        
    }
    //BLOCK_END Object_To_World_Dir

    //BLOCK_BEGIN Edge_AA_Vertex 342

    void Edge_AA_Vertex_B342(
        float3 Position_World,
        float3 Normal_World,
        float3 Position_Object,
        float3 Normal_Object,
        float3 Eye,
        float Radial_Gradient,
        float3 Radial_Dir,
        float3 Tangent,
        out float Gradient1,
        out float Gradient2    )
    {
        // main code goes here
        float3 I = (Eye-Position_World);
        if (Normal_Object.z==0) { // edge
            float3 T = Position_Object.z>0.0 ? float3(0.0,0.0,1.0) : float3(0.0,0.0,-1.0);
            T = (mul((float3x3)unity_ObjectToWorld, Tangent));
            float g = (dot(T,I)<0.0) ? 0.0 : 1.0;
            Gradient1 = Position_Object.z>0.0 ? g : 1.0;
            Gradient2 = Position_Object.z>0.0 ? 1.0 : g;
        } else {
            float3 R = (mul((float3x3)unity_ObjectToWorld, Tangent)); //Radial_Dir);
            float k = (dot(R,I)>0.0 ? 1.0 : 0.0);
        //    float kk = dot(normalize(R),normalize(I));
        //    float k =  kk>0.0 ? kk*Edge_Bend : 0.0;
            Gradient1 = k + (1.0-k)*(Radial_Gradient);
            Gradient2 = 1.0;
        }
        
    }
    //BLOCK_END Edge_AA_Vertex


    VertexOutput vert(VertexInput vertInput)
    {
        UNITY_SETUP_INSTANCE_ID(vertInput);
        VertexOutput o;
        UNITY_INITIALIZE_VERTEX_OUTPUT_STEREO(o);


        // Object_To_World_Dir
        float3 Tangent_World_Q302;
        Tangent_World_Q302 = (mul((float3x3)unity_ObjectToWorld, float3(1,0,0)));
        
        float3 Binormal_World_Q304;
        Object_To_World_Dir_B304(float3(0,1,0),Binormal_World_Q304);

        // Anisotropy
        float Anisotropy_Q316=length(Tangent_World_Q302)/length(Binormal_World_Q304);

        // Object_To_World_Normal
        float3 Nrm_World_Q301;
        Nrm_World_Q301 = normalize(UnityObjectToWorldNormal(vertInput.normal));
        
        // Normalize3
        float3 Tangent_World_N_Q305 = normalize(Tangent_World_Q302);

        // Normalize3
        float3 Binormal_World_N_Q306 = normalize(Binormal_World_Q304);

        float3 Pos_World_Q312;
        Object_To_World_Pos_B300(_Iridescence_Center_,Pos_World_Q312);

        float3 New_P_Q325;
        float2 New_UV_Q325;
        float Radial_Gradient_Q325;
        float3 Radial_Dir_Q325;
        Move_Verts_B325(Anisotropy_Q316,vertInput.vertex.xyz,_Line_Width_,New_P_Q325,New_UV_Q325,Radial_Gradient_Q325,Radial_Dir_Q325);

        // Incident3
        float3 Incident_Q308 = normalize(Pos_World_Q312 - _WorldSpaceCameraPos);

        float3 Pos_World_Q300;
        Object_To_World_Pos_B300(New_P_Q325,Pos_World_Q300);

        float4 Blob_Info_Q329;
        if (_Blob_Enable_) {
          Blob_Vertex_B329(Pos_World_Q300,Nrm_World_Q301,Tangent_World_N_Q305,Binormal_World_N_Q306,_Blob_Position_,_Blob_Intensity_,_Blob_Near_Size_,_Blob_Far_Size_,_Blob_Near_Distance_,_Blob_Far_Distance_,_Blob_Fade_Length_,_Blob_Pulse_,_Blob_Fade_,Blob_Info_Q329);
        } else {
          Blob_Info_Q329 = float4(0,0,0,0);
        }

        float4 Blob_Info_Q330;
        if (_Blob_Enable_2_) {
          Blob_Vertex_B330(Pos_World_Q300,Nrm_World_Q301,Tangent_World_N_Q305,Binormal_World_N_Q306,_Blob_Position_2_,_Blob_Intensity_,_Blob_Near_Size_2_,_Blob_Far_Size_,_Blob_Near_Distance_,_Blob_Far_Distance_,_Blob_Fade_Length_,_Blob_Pulse_2_,_Blob_Fade_2_,Blob_Info_Q330);
        } else {
          Blob_Info_Q330 = float4(0,0,0,0);
        }

        // DotProduct3
        float Dot_Q311;
        Dot_Q311 = dot(Tangent_World_N_Q305, Incident_Q308);
        //Dot_Q311 = sign(Dot_Q311)*Dot_Q311*Dot_Q311;
        
        float Gradient1_Q342;
        float Gradient2_Q342;
        if (_Smooth_Edges_) {
          Edge_AA_Vertex_B342(Pos_World_Q300,Nrm_World_Q301,vertInput.vertex.xyz,vertInput.normal,_WorldSpaceCameraPos,Radial_Gradient_Q325,Radial_Dir_Q325,vertInput.tangent,Gradient1_Q342,Gradient2_Q342);
        } else {
          Gradient1_Q342 = 1;
          Gradient2_Q342 = 1;
        }

        float3 Iridescence_Q327;
        Iridescence_Vertex_B327(_Iridescence_Intensity_,_Iridescence_Ramp_,_Left_X0_,_Left_X1_,_Right_X0_,_Right_X1_,_Angle_,New_UV_Q325,Dot_Q311,Iridescence_Q327);

        float2 Rect_UV_Q345;
        float4 Rect_Parms_Q345;
        float2 Scale_XY_Q345;
        Round_Rect_Vertex_B345(New_UV_Q325,Tangent_World_Q302,Binormal_World_Q304,_Line_Width_,Anisotropy_Q316,Gradient1_Q342,Gradient2_Q342,Rect_UV_Q345,Rect_Parms_Q345,Scale_XY_Q345);

        float3 Line_Vertex_1_Q340;
        float3 Line_Vertex_2_Q340;
        Line_Vertex_B340(Scale_XY_Q345,_Draw_,_Draw_Fuzz_,_Draw_Start_,Rect_UV_Q345,_Time.y,_Rate_,_Highlight_Transform_,Line_Vertex_1_Q340,Line_Vertex_2_Q340);

        float3 Position = Pos_World_Q300;
        float3 Normal = Iridescence_Q327;
        float2 UV = Rect_UV_Q345;
        float3 Tangent = Line_Vertex_1_Q340;
        float3 Binormal = Line_Vertex_2_Q340;
        float4 Color = float4(1,1,1,1);
        float4 Extra1 = Rect_Parms_Q345;
        float4 Extra2 = Blob_Info_Q329;
        float4 Extra3 = Blob_Info_Q330;


        o.pos = UnityObjectToClipPos(vertInput.vertex);
        o.pos = mul(UNITY_MATRIX_VP, float4(Position,1));
        o.normalWorld.xyz = Normal; o.normalWorld.w=1.0;
        o.uv = UV;
        o.tangent.xyz = Tangent; o.tangent.w=1.0;
        o.binormal.xyz = Binormal; o.binormal.w=1.0;
        o.extra1=Extra1;
        o.extra2=Extra2;
        o.extra3=Extra3;

        return o;
    }

    //BLOCK_BEGIN Round_Rect_Fragment 344

    void Round_Rect_Fragment_B344(
        half Line_Width,
        half4 Line_Color,
        half Filter_Width,
        half2 UV,
        half Line_Visibility,
        half4 Rect_Parms,
        half4 Fill_Color,
        out half4 Color    )
    {
        float g = min(Rect_Parms.z,Rect_Parms.w);
        float dgrad = max(fwidth(g)*Filter_Width,0.00001);
        float Inside_Rect = saturate(g/dgrad);
        
        float2 absUV = abs(UV);
        float2 dUV = max(fwidth(absUV),0.00001);
        float2 g2 = 1.0-saturate((absUV-Rect_Parms.xy)/dUV);
        float inner = g2.x*g2.y;
        
        //float d = length(max(abs(UV)-Rect_Parms.xy,0.0));
        //float dx = max(fwidth(d)*Filter_Width,0.00001);
        
        //this is arguably more correct...
        //float inner = saturate((d+dx*0.5-max(Rect_Parms.z,d-dx*0.5))/dx);
        //float inner = saturate((d+dx*0.5-max(Radius-Line_Width,d-dx*0.5))/dx);
        
        Color = saturate(lerp(Line_Color*Line_Visibility, Fill_Color,float4( inner, inner, inner, inner)))*Inside_Rect;
        //but this saves 3 ops
        //float inner = saturate((Rect_Parms.z-d)/dx);
        //Color = lerp(Line_Color*Line_Visibility, Fill_Color,float4( inner, inner, inner, inner))*Inside_Rect;
    }
    //BLOCK_END Round_Rect_Fragment

    //BLOCK_BEGIN Line_Fragment 335

    void Line_Fragment_B335(
        float4 Base_Color,
        float4 Highlight_Color,
        half Highlight_Width,
        half3 Line_Vertex_1,
        half3 Line_Vertex_2,
        half Draw_Fuzz_Inv,
        out half In_Line,
        out float4 Line_Color    )
    {
        half k = dot(Line_Vertex_1.xy,float2(abs(Line_Vertex_2.y),Line_Vertex_2.z));
        In_Line = saturate((k-Line_Vertex_1.z)*Draw_Fuzz_Inv);
        
        half k2 = saturate(abs(Line_Vertex_2.y/Highlight_Width));
        Line_Color = lerp(Highlight_Color,Base_Color,float4(k2,k2,k2,k2));
    }
    //BLOCK_END Line_Fragment

    //BLOCK_BEGIN Blob_Fragment 333

    void Blob_Fragment_B333(
        sampler2D Blob_Texture,
        float4 Blob_Info,
        out float4 Blob_Color    )
    {
        half k = dot(Blob_Info.xy,Blob_Info.xy);
        Blob_Color = Blob_Info.w * tex2D(Blob_Texture,float2(float2(sqrt(k),Blob_Info.z).x,1.0-float2(sqrt(k),Blob_Info.z).y))*saturate(1.0-k);
        
    }
    //BLOCK_END Blob_Fragment


    //fixed4 frag(VertexOutput fragInput, fixed facing : VFACE) : SV_Target
    half4 frag(VertexOutput fragInput) : SV_Target
    {
        half4 result;

        // Divide
        half Quotient_Q336 = 1 / _Draw_Fuzz_;

        // Add_Colors
        float4 Base_And_Iridescent_Q314;
        Base_And_Iridescent_Q314 = _Base_Color_ + float4(fragInput.normalWorld.xyz,0.0);
        
        float4 Blob_Color_Q333;
        if (_Blob_Enable_) {
          Blob_Fragment_B333(_Blob_Texture_,fragInput.extra2,Blob_Color_Q333);
        } else {
          Blob_Color_Q333 = float4(0,0,0,0);
        }

        float4 Blob_Color_Q334;
        if (_Blob_Enable_2_) {
          Blob_Fragment_B333(_Blob_Texture_,fragInput.extra3,Blob_Color_Q334);
        } else {
          Blob_Color_Q334 = float4(0,0,0,0);
        }

        half In_Line_Q335;
        float4 Line_Color_Q335;
        Line_Fragment_B335(_Line_Color_,_Highlight_Color_,_Highlight_Width_,fragInput.tangent.xyz,fragInput.binormal.xyz,Quotient_Q336,In_Line_Q335,Line_Color_Q335);

        // Combine_Blobs
        float4 Blobs_Q337;
        Blobs_Q337 = Blob_Color_Q333+Blob_Color_Q334;
        Blobs_Q337=min(Blobs_Q337,float4(1.0,1.0,1.0,1.0));
        
        // Blend_Over
        float4 Result_Q323 = Blobs_Q337 + (1.0 - Blobs_Q337.a) * Base_And_Iridescent_Q314;

        half4 Color_Q344;
        Round_Rect_Fragment_B344(_Line_Width_,Line_Color_Q335,_Filter_Width_,fragInput.uv,In_Line_Q335,fragInput.extra1,Result_Q323,Color_Q344);

        // Scale_Color
        float4 Result_Q338 = _Fade_Out_ * Color_Q344;

        float4 Out_Color = Result_Q338;
        float Clip_Threshold = 0.001;
        bool To_sRGB = false;

        result = Out_Color;
        return result;
    }

    ENDCG
  }
 }
}
