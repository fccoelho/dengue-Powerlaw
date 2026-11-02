const pptxgen = require("pptxgenjs");

const slideConfig = {
  type: 'summary',
  index: 12,
  title: 'Thank You'
};

function createSlide(pres, theme) {
  const slide = pres.addSlide();
  slide.background = { color: theme.primary };

  slide.addText("Thank You", {
    x: 0.5, y: 1.8, w: 9, h: 1.0,
    fontSize: 48, fontFace: "Arial",
    color: "FFFFFF", bold: true, align: "center"
  });

  slide.addShape(pres.shapes.RECTANGLE, {
    x: 3.5, y: 2.9, w: 3, h: 0.05,
    fill: { color: theme.accent }
  });

  slide.addText("Power Laws and Prediction Gaps", {
    x: 0.5, y: 3.2, w: 9, h: 0.6,
    fontSize: 20, fontFace: "Arial",
    color: theme.light, align: "center"
  });

  slide.addText("A Statistical Framework for Extreme Dengue Outbreaks in Brazilian Cities", {
    x: 0.5, y: 3.7, w: 9, h: 0.5,
    fontSize: 14, fontFace: "Arial",
    color: theme.light, align: "center"
  });

  const keywords = () => {
    const kws = ["Power Law", "Dengue", "Extreme Events", "Machine Learning", "Spatial Analysis"];
    slide.addText(kws.join("  •  "), {
      x: 0.5, y: 4.5, w: 9, h: 0.4,
      fontSize: 12, fontFace: "Arial",
      color: theme.accent, align: "center"
    });
  };
  keywords();

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
  pres.writeFile({ fileName: "slide-12-preview.pptx" });
}

module.exports = { createSlide, slideConfig };
