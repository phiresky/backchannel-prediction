const webpack = require('webpack');
const fs = require('fs');
module.exports = {
    entry: './plot.tsx',
    devtool: 'source-map',
    plugins: [
        new webpack.DefinePlugin({
            "VERSIONS": JSON.stringify(fs.readdirSync("../../trainNN/out"))
        })
    ],
    output: {
        filename: 'dist/plot.js',
        publicPath: "evaluate/plot"
    },
    module: {
        rules: [
            { test: /\.tsx?$/, loader: 'awesome-typescript-loader' }
        ]
    }
}