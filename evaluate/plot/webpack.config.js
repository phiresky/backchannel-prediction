const webpack = require('webpack');
const fs = require('fs');
const path = require('path');
const Html = require('html-webpack-plugin');
const ExtractTextPlugin = require("extract-text-webpack-plugin");

module.exports = {
    entry: './plot',
    devtool: 'source-map',
    plugins: [
        new Html({
            title: 'Backchanneler NN evaluation'
        }),
        new ExtractTextPlugin("styles.[hash].css"),
        new webpack.DefinePlugin({
            'process.env': {
                'NODE_ENV': JSON.stringify('production')
            }
        }),
        new webpack.DefinePlugin({
            "VERSIONS": JSON.stringify(fs.readdirSync("../../trainNN/out"))
        })
    ],
    output: {
        path: path.resolve('./dist'),
        filename: 'plot.[hash].js',
        publicPath: "./"
    },
    resolve: { extensions: [".js", ".json", ".ts", ".tsx"] },
    module: {
        rules: [
            { test: /\.tsx?$/, loader: 'babel-loader?presets=es2015!awesome-typescript-loader' },
            { test: /\.css$/, loader: ExtractTextPlugin.extract('css-loader') },
            { test: /\.less$/, loader: ExtractTextPlugin.extract('css-loader!less-loader') }
        ]
    },
    devServer: {
        publicPath: "/evaluate/plot/dist"
    }
}