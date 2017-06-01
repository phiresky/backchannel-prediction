var path = require('path');
var webpack = require('webpack');
const ExtractTextPlugin = require('extract-text-webpack-plugin');
const autoprefixer = require('autoprefixer');

module.exports = {
  context: __dirname,
  devtool: 'source-map',
  entry: [
    './client'
  ],
  output: {
    path: path.join(__dirname, 'dist'),
    publicPath: '/dist',
    filename: 'bundle.js',
  },
  resolve: {
    extensions: ['', '.scss', '.css', '.js', '.ts', '.tsx'],
    packageMains: ['browser', 'web', 'browserify', 'main', 'style'],
  },
  module: {
    loaders: [{
      test: /\.tsx?$/,
      loaders: ['babel-loader', 'ts-loader'],
      //include: path.join(__dirname, 'src')
    },
    {
      test: /(\.scss)$/,
      loader: ExtractTextPlugin.extract('style', 'css?sourceMap&modules&importLoaders=1&localIdentName=[name]__[local]___[hash:base64:5]!postcss!sass')
    },
    { test: /\.css$/, loader: "style-loader!css-loader" },
    ]
  },
  postcss: [autoprefixer],
  sassLoader: {
    data: '@import "theme/_config.scss";',
    includePaths: [path.resolve(__dirname, './src')]
  },
  ts: {
    transpileOnly: true
  },
  plugins: [
    new webpack.DefinePlugin({
      'process.env.NODE_ENV': JSON.stringify('development')
    }),
    new ExtractTextPlugin('bundle.css', { allChunks: true }),
    new webpack.optimize.OccurenceOrderPlugin(),
    new webpack.NoErrorsPlugin(),
    new webpack.HotModuleReplacementPlugin(),
  ]
};
