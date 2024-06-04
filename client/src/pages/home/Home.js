import React, { useEffect } from "react";
import "./home.css"
import { useState } from "react";
import { useForm } from "react-hook-form";
import { yupResolver } from '@hookform/resolvers/yup';
import * as yup from 'yup';
import axios from 'axios'

const rectangleSchema = yup.object().shape({
    width: yup
        .number()
        .typeError('Width must be a number')
        .positive('Width must be greater than 0')
        .required('Width is required'),
    height: yup
        .number()
        .typeError('Height must be a number')
        .positive('Height must be greater than 0')
        .required('Height is required'),
    t_top: yup
        .number()
        .typeError('Top thickness must be a number')
        .min(0, 'Top thickness must be at least 0'),
    t_right: yup
        .number()
        .typeError('Right thickness must be a number')
        .min(0, 'Right thickness must be at least 0'),
    t_bottom: yup
        .number()
        .typeError('Bottom thickness must be a number')
        .min(0, 'Bottom thickness must be at least 0'),
    t_left: yup
        .number()
        .typeError('Left thickness must be a number')
        .min(0, 'Left thickness must be at least 0')
}).test(
    'thicknessCheck',
    'Top + Bottom thickness should not exceed height and Left + Right thickness should not exceed width',
    function (values) {
        const { width, height, t_top, t_right, t_bottom, t_left } = values;
        const isValidHeight = (t_top + t_bottom) <= height;
        const isValidWidth = (t_left + t_right) <= width;
        return isValidHeight && isValidWidth;
    }
);

const circleSchema = yup.object().shape({
    radius: yup
        .number()
        .typeError('Radius must be a number')
        .positive('Radius must be greater than 0')
        .required('Radius is required'),
    thickness: yup
        .number()
        .typeError('Thickness must be a number')
        .min(0, 'Thickness must be at least 0')
        .test(
            'thicknessCheck',
            'Thickness should not exceed half of the radius',
            function (value) {
                return value <= this.parent.radius / 2;
            }
        )
});

const frameSchema = yup.object().shape({
    frameWidth: yup
      .number()
      .required("Width is required")
      .min(0, "Width must be greater than or equal to 0"),
    frameHeight: yup
      .number()
      .required("Height is required")
      .min(0, "Height must be greater than or equal to 0"),
  });

const Home = () => {

    const [nestImages, setNestImages] = useState([]);
    const { register: registerDimensions, handleSubmit : handleSubmitDimensions, reset: resetDimensions } = useForm();
    const { register: registerFrame, handleSubmit: handleSubmitFrame, reset: resetFrame } = useForm();
    const [selectedButton, setSelectedButton] = useState('square');
    const [dimensions, setDimensions] = useState([]);
    const [frames, setFrames] = useState();

    const rectangleForm = useForm({
        resolver: yupResolver(rectangleSchema),
        defaultValues: {
            width: '',
            height: '',
            t_top: '',
            t_right: '',
            t_bottom: '',
            t_left: ''
        }
    });

    const circleForm = useForm({
        resolver: yupResolver(circleSchema),
        defaultValues: {
            radius: '',
            thickness: ''
        }
    });

    const frameForm = useForm({
        resolver: yupResolver(frameSchema),
        defaultValues: {
            frameWidth: 0,
            frameHeight: 0
        }
    });

    const handleButtonClick = (buttonType) => {
        setSelectedButton(buttonType);
        setDimensions([]);
        setFrames([]);
        resetDimensions();
        resetFrame();
    };

    const onSubmitDimensions = (data) => {
        if (selectedButton === 'square') {
            const newDimensions = {
                type: 'square',
                width: parseFloat(data.width),
                height: parseFloat(data.height),
                t_top : parseFloat(data.t_top),
                t_right : parseFloat(data.t_right),
                t_bottom : parseFloat(data.t_bottom),
                t_left : parseFloat(data.t_left)
            };
            setDimensions([...dimensions, newDimensions]);
        } else if (selectedButton === 'circle') {
            const newDimensions = {
                type: 'circle',
                radius: parseFloat(data.radius),
                thickness: parseFloat(data.thickness)
            };
            setDimensions([...dimensions, newDimensions]);
        }
        resetDimensions();
    };

    const onSubmitFrames = (data) => {
        const newFrame = {
            width: parseFloat(data.frameWidth),
            height: parseFloat(data.frameHeight)
        };
        setFrames(newFrame);
        console.log(frames)
        resetFrame();
    };

    function handleClear() {
        const plotBarDiv = document.querySelector('.plot_bar');
        if (plotBarDiv) {
            while (plotBarDiv.firstChild) {
                plotBarDiv.removeChild(plotBarDiv.firstChild);
            }
        }
    }

    function displayImage(base64String) {
        console.log(nestImages)
        const byteCharacters = atob(base64String);
        const byteNumbers = new Array(byteCharacters.length);
        for (let i = 0; i < byteCharacters.length; i++) {
          byteNumbers[i] = byteCharacters.charCodeAt(i);
        }
        const byteArray = new Uint8Array(byteNumbers);
        const blob = new Blob([byteArray], { type: 'image/png' });
      
        const imageUrl = URL.createObjectURL(blob);

        const imgElement = document.createElement('img');
        imgElement.src = imageUrl;
        console.log(imageUrl)

        const plotBarDiv = document.querySelector('.plot_bar');
        if (plotBarDiv) {
          plotBarDiv.appendChild(imgElement);
        }
        setNestImages([])
    }

    useEffect(() => {
        if (nestImages && nestImages.length > 0) {
            console.log(nestImages)
            displayImage(nestImages);
        }
    }, [nestImages]);

    const handleNest = async () => {
        let postData;
        if (selectedButton === 'square') {
          postData = {
            type: 'rectangle',
            dimensions,
            frames
          };
        } else if (selectedButton === 'circle') {
          postData = {
            type: 'circle',
            dimensions,
            frames
          };
        }

        const s = postData;

        try {
            const response = await axios.post('http://localhost:8000/nest', s, {
                headers: {
                    'Content-Type': 'application/json'
                }
            });
            setNestImages(response.data.image);
        } catch (error) {
            console.error('Error nesting shapes:', error);
        }
    };
    
    return(
        <div className="home">
            <div className="container">
                <div className="nav_bar">
                    <button
                        className={`square-btn ${selectedButton === 'square' ? 'selected' : ''}`}
                        onClick={() => handleButtonClick('square')}
                    >
                        Squares
                    </button>
                    <button
                        className={`circle-btn ${selectedButton === 'circle' ? 'selected' : ''}`}
                        onClick={() => handleButtonClick('circle')}
                    >
                        Circles
                    </button>
                </div>
                <div className="input_bar">
                    {selectedButton === 'square' && (
                        <form onSubmit={rectangleForm.handleSubmit(onSubmitDimensions)}>
                            <div>
                                <div className="title-input">Rectangles</div>
                                <div className="data">
                                    <input
                                        placeholder="Width"
                                        className="input-data"
                                        {...rectangleForm.register("width")}
                                    />
                                    {rectangleForm.formState.errors.width && (
                                        <span className="error">{rectangleForm.formState.errors.width.message}</span>
                                    )}
                                </div>
                                <div className="data">
                                    <input
                                        placeholder="Height"
                                        className="input-data"
                                        {...rectangleForm.register("height")}
                                    />
                                    {rectangleForm.formState.errors.height && (
                                        <span className="error">{rectangleForm.formState.errors.height.message}</span>
                                    )}
                                </div>
                                <div className="data">
                                    <input
                                        placeholder="Top Thickness"
                                        className="input-data"
                                        defaultValue={0}
                                        {...rectangleForm.register("t_top")}
                                    />
                                    {rectangleForm.formState.errors.t_top && (
                                        <span className="error">{rectangleForm.formState.errors.t_top.message}</span>
                                    )}
                                </div>
                                <div className="data">
                                    <input
                                        placeholder="Right Thickness"
                                        className="input-data"
                                        defaultValue={0}
                                        {...rectangleForm.register("t_right")}
                                    />
                                    {rectangleForm.formState.errors.t_right && (
                                        <span className="error">{rectangleForm.formState.errors.t_right.message}</span>
                                    )}
                                </div>
                                <div className="data">
                                    <input
                                        placeholder="Bottom Thickness"
                                        className="input-data"
                                        defaultValue={0}
                                        {...rectangleForm.register("t_bottom")}
                                    />
                                    {rectangleForm.formState.errors.t_bottom && (
                                        <span className="error">{rectangleForm.formState.errors.t_bottom.message}</span>
                                    )}
                                </div>
                                <div className="data">
                                    <input
                                        placeholder="Left Thickness"
                                        className="input-data"
                                        defaultValue={0}
                                        {...rectangleForm.register("t_left")}
                                    />
                                    {rectangleForm.formState.errors.t_left && (
                                        <span className="error">{rectangleForm.formState.errors.t_left.message}</span>
                                    )}
                                </div>
                                <button type="submit" className="save-btn">
                                    Save Dimensions
                                </button>
                                <div className="dimensions-list">
                                    {dimensions
                                        .filter(dim => dim.type === 'square')
                                        .map((dim, index) => (
                                            <div key={index}>
                                                r{index + 1} = (w: {dim.width}, h: {dim.height})
                                                <br/>
                                                t{index + 1} = (t: {dim.t_top}, l: {dim.t_left}, b: {dim.t_bottom}, r: {dim.t_right})
                                            </div>
                                        ))}
                                </div>
                            </div>
                        </form>
                    )}
                    {selectedButton === 'circle' && (
                        <form onSubmit={circleForm.handleSubmit(onSubmitDimensions)}>
                            <div>
                                <div className="title-input">Circles</div>
                                <div className="data">
                                    <input
                                        placeholder="Radius"
                                        className="input-data"
                                        {...circleForm.register("radius")}
                                    />
                                    {circleForm.formState.errors.radius && (
                                        <span className="error">{circleForm.formState.errors.radius.message}</span>
                                    )}
                                </div>
                                <div className="data">
                                    <input
                                        placeholder="Thickness"
                                        className="input-data"
                                        defaultValue={0}
                                        {...circleForm.register("thickness")}
                                    />
                                    {circleForm.formState.errors.thickness && (
                                        <span className="error">{circleForm.formState.errors.thickness.message}</span>
                                    )}
                                </div>
                                <button type="submit" className="save-btn">
                                    Save Radius
                                </button>
                                <div className="dimensions-list">
                                    {dimensions
                                        .filter(dim => dim.type === 'circle')
                                        .map((dim, index) => (
                                            <div key={index}>
                                                c{index + 1} = (r: {dim.radius}, t: {dim.thickness})
                                            </div>
                                        ))}
                                </div>
                            </div>
                        </form>
                    )}
                    <div>
                        <div className="title-input">Frame</div>
                        <form onSubmit={frameForm.handleSubmit(onSubmitFrames)}>
                            <div className="data">
                                <input
                                    placeholder="Frame Width"
                                    className="input-data"
                                    {...frameForm.register("frameWidth")}
                                />
                                {frameForm.formState.errors.frameWidth && (
                                    <span className="error">{frameForm.formState.errors.frameWidth.message}</span>
                                )}
                            </div>
                            <div className="data">
                                <input
                                    placeholder="Frame Height"
                                    className="input-data"
                                    {...frameForm.register("frameHeight")}
                                />
                                {frameForm.formState.errors.frameHeight && (
                                    <span className="error">{frameForm.formState.errors.frameHeight.message}</span>
                                )}
                            </div>
                            <button type="submit" className="save-btn">
                                Save Frame
                            </button>
                            <div className="dimensions-list">
                                    {frames && <div>
                                        f =  (w: {frames.width}, h: {frames.height})
                                        </div>
                                    }
                            </div>
                        </form>
                    </div>
                    <button onClick={handleNest} className="nest-btn">
                        Nest
                    </button>
                    <button onClick={handleClear} className="nest-btn">
                        Clear
                    </button>
                </div>
                <div className="plot_bar">
                </div>
            </div>
        </div>
    );
}

export default Home;