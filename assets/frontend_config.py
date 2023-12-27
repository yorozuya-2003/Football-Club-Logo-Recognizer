import base64
import streamlit as st


def set_bg_image(bg_img, bg_img_ext):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: url(data:image/{bg_img_ext};base64,{base64.b64encode(open(bg_img, "rb").read()).decode()});
            background-size: cover
        }}
        </style>
        """,
        unsafe_allow_html=True
    )


def custom_title(component, txt):
    html_str = f'''
        <style>
        #animated-header {{
            background: linear-gradient(
                to right,
                #ffcc99 20%,  /* Light Orange */
                #ffe066 30%,  /* Light Yellow */
                #ffb3e6 70%,  /* Light Pink */
                #99ccff 80%   /* Light Blue */
            );
            -webkit-background-clip: text;
            background-clip: text;
            -webkit-text-fill-color: transparent;
            text-fill-color: transparent;
            background-size: 500% auto;
            animation: textShine 2.5s ease-in-out infinite alternate;
            font-size: 2em;
            font-family: "Montserrat", sans-serif;

        }}

        @keyframes textShine {{
            0% {{
                background-position: 0% 50%;
            }}
            100% {{
                background-position: 100% 50%;
            }}
        }}
        </style>
        <p id="animated-header">
            {txt}
        </p>
    '''
    component.markdown(html_str, unsafe_allow_html=True)


def custom_image_caption(component, txt):
    component.caption(f"""
        <p
            style='text-align:center;
            font-family: "Montserrat",
            sans-serif; background-color:rgb(38, 39, 48);
            color:white; border-radius:20px;
            font-size:20px;
            opacity:0.8;'
        >
            {txt}
        </p>
    """, unsafe_allow_html=True)


def custom_success_message(component, txt):
    html_str = f"""
        <p
            style='background-color:rgb(38, 39, 48);
            color:white;
            font-size:18px;
            border-radius:0.5rem;
            line-height:60px;
            text-align:center;
            margin:0px;
            opacity:0.8;
            font-family: "Montserrat", sans-serif;
            border-style: solid;
            border-color: white;
            outline-width: 10px;'
        >
            {txt}
        </style><br></p>
    """
    component.markdown(html_str, unsafe_allow_html=True) 