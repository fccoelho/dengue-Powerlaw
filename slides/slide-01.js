const pptxgen = require("pptxgenjs");

const slideConfig = {
  type: 'cover',
  index: 1,
  title: 'Power Laws and Prediction Gaps'
};

function createSlide(pres, theme) {
  const slide = pres.addSlide();
  slide.background = { color: theme.primary };

  slide.addText("Power Laws and Prediction Gaps", {
    x: 0.5, y: 1.8, w: 9, h: 1.2,
    fontSize: 40, fontFace: "Arial",
    color: "FFFFFF", bold: true, align: "center"
  });

  slide.addText("A Statistical Framework for Extreme Dengue Outbreaks\nin Brazilian Cities", {
    x: 0.5, y: 3.0, w: 9, h: 0.9,
    fontSize: 22, fontFace: "Arial",
    color: theme.light, align: "center"
  });

  slide.addShape(pres.shapes.RECTANGLE, {
    x: 3.5, y: 4.2, w: 3, h: 0.05,
    fill: { color: theme.accent }
  });

  slide.addText("15 Years of Epidemic Data | 5,000+ Municipalities", {
    x: 0.5, y: 4.5, w: 9, h: 0.5,
    fontSize: 14, fontFace: "Arial",
    color: theme.light, align: "center"
  });

  return slide;
}

if (require.main === module) {
  const pres = new pptxgen();
  pres.layout = 'LAYOUT_16x9';
  const theme = {
    primary: "1A365D",
    secondary: "2C5282",
    accent: "4ECDC4",
    light: "E2E8F0",
    bg: "F7FAFC"
  };
  createSlide(pres, theme);
  pres.writeFile({ fileName: "slide-01-preview.pptx" });
}

module.exports = { createSlide, slideConfig };
