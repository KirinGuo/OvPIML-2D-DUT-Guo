%% Calculate the derivation of shape function and Jacobi matrix
function [wN,dwNdx] = Shape_weightFunction(GaussPoint, Jacobi, type)
ParentNodes=[-1  1  1 -1;
    -1 -1  1  1]; %The coordinates for the reference element
switch type
    case 1
        wN=zeros(1,4);
        ParentNDerivative=zeros(2,4);
        for I=1:4
            XPoint = ParentNodes(1,I);
            YPoint = ParentNodes(2,I);
            ShapePart = [1+GaussPoint(1)*XPoint 1+GaussPoint(2)*YPoint];
            wN(I)=0.25*ShapePart(1)*ShapePart(2);
            ParentNDerivative(1,I) = 0.25*XPoint*ShapePart(2);
            ParentNDerivative(2,I) = 0.25*YPoint*ShapePart(1);
        end
    case 2
        wN=zeros(1,4);
        ParentNDerivative=zeros(2,4);     
        for I=1:4
            XPoint = ParentNodes(1,I);
            YPoint = ParentNodes(2,I);
            ShapePart = [1+(GaussPoint(1)-1)/2*XPoint 1+GaussPoint(2)*YPoint];
            wN(I)=0.25*ShapePart(1)*ShapePart(2);
            ParentNDerivative(1,I) = 0.125*XPoint*ShapePart(2);
            ParentNDerivative(2,I) = 0.25*YPoint*ShapePart(1);
        end
    case 3
        wN=zeros(1,4);
        ParentNDerivative=zeros(2,4);        
        for I=1:4
            XPoint = ParentNodes(1,I);
            YPoint = ParentNodes(2,I);
            ShapePart = [1+(GaussPoint(1)+1)/2*XPoint 1+GaussPoint(2)*YPoint];
            wN(I)=0.25*ShapePart(1)*ShapePart(2);
            ParentNDerivative(1,I) = 0.125*XPoint*ShapePart(2);
            ParentNDerivative(2,I) = 0.25*YPoint*ShapePart(1);
        end
    case 4
        wN=zeros(1,4);
        ParentNDerivative=zeros(2,4);        
        for I=1:4
            XPoint = ParentNodes(1,I);
            YPoint = ParentNodes(2,I);
            ShapePart = [1+GaussPoint(1)*XPoint 1+(GaussPoint(2)+1)/2*YPoint];
            wN(I)=0.25*ShapePart(1)*ShapePart(2);
            ParentNDerivative(1,I) = 0.25*XPoint*ShapePart(2);
            ParentNDerivative(2,I) = 0.125*YPoint*ShapePart(1);
        end
    case 5
        wN=zeros(1,4);
        ParentNDerivative=zeros(2,4);        
        for I=1:4
            XPoint = ParentNodes(1,I);
            YPoint = ParentNodes(2,I);
            ShapePart = [1+GaussPoint(1)*XPoint 1+(GaussPoint(2)-1)/2*YPoint];
            wN(I)=0.25*ShapePart(1)*ShapePart(2);
            ParentNDerivative(1,I) = 0.25*XPoint*ShapePart(2);
            ParentNDerivative(2,I) = 0.125*YPoint*ShapePart(1);
        end
    case 6
        wN=zeros(1,8);
        ParentNDerivative=zeros(2,8);
        for I=1:4
            XPoint = ParentNodes(1,I);
            YPoint = ParentNodes(2,I);
            ShapePart = [1+GaussPoint(2)*YPoint 1+(GaussPoint(1)+1)/2*XPoint 1+(GaussPoint(1)-1)/2*XPoint];
            wN(I)=0.25*ShapePart(2)*ShapePart(1)*(1-GaussPoint(1))/2;
            wN(I+4)=0.25*ShapePart(3)*ShapePart(1)*(1+GaussPoint(1))/2;
            ParentNDerivative(1,I) = 0.125*XPoint*ShapePart(1)*(1-GaussPoint(1))/2 - 0.125*ShapePart(2)*ShapePart(1);
            ParentNDerivative(2,I) = 0.25*YPoint*ShapePart(2)*(1-GaussPoint(1))/2;
            ParentNDerivative(1,I+4) = 0.125*XPoint*ShapePart(1)*(1+GaussPoint(1))/2 + 0.125*ShapePart(3)*ShapePart(1);
            ParentNDerivative(2,I+4) = 0.25*YPoint*ShapePart(3)*(1+GaussPoint(1))/2;
        end
    case 7
        wN=zeros(1,8);
        ParentNDerivative=zeros(2,8);
        for I=1:4
            XPoint = ParentNodes(1,I);
            YPoint = ParentNodes(2,I);
            ShapePart = [1+GaussPoint(1)*XPoint 1+(GaussPoint(2)-1)/2*YPoint 1+(GaussPoint(2)+1)/2*YPoint];
            wN(I)=0.25*ShapePart(1)*ShapePart(2)*(1+GaussPoint(2))/2;
            wN(I+4)=0.25*ShapePart(1)*ShapePart(3)*(1-GaussPoint(2))/2;
            ParentNDerivative(1,I) = 0.25*XPoint*ShapePart(2)*(1+GaussPoint(2))/2;
            ParentNDerivative(2,I) = 0.125*YPoint*ShapePart(1)*(1+GaussPoint(2))/2 + 0.125*ShapePart(1)*ShapePart(2);
            ParentNDerivative(1,I+4) = 0.25*XPoint*ShapePart(3)*(1-GaussPoint(2))/2;
            ParentNDerivative(2,I+4) = 0.125*YPoint*ShapePart(1)*(1-GaussPoint(2))/2 - 0.125*ShapePart(1)*ShapePart(3);
        end        
    case 8
        wN=zeros(1,4);
        ParentNDerivative=zeros(2,4);        
        for I=1:4
            XPoint = ParentNodes(1,I);
            YPoint = ParentNodes(2,I);
            ShapePart = [1+(GaussPoint(1)-1)/2*XPoint 1+(GaussPoint(2)+1)/2*YPoint];
            wN(I)=0.25*ShapePart(1)*ShapePart(2);
            ParentNDerivative(1,I) = 0.125*XPoint*ShapePart(2);
            ParentNDerivative(2,I) = 0.125*YPoint*ShapePart(1);
        end
    case 9
        wN=zeros(1,4);
        ParentNDerivative=zeros(2,4);        
        for I=1:4
            XPoint = ParentNodes(1,I);
            YPoint = ParentNodes(2,I);
            ShapePart = [1+(GaussPoint(1)+1)/2*XPoint 1+(GaussPoint(2)+1)/2*YPoint];
            wN(I)=0.25*ShapePart(1)*ShapePart(2);
            ParentNDerivative(1,I) = 0.125*XPoint*ShapePart(2);
            ParentNDerivative(2,I) = 0.125*YPoint*ShapePart(1);
        end
    case 10
        wN=zeros(1,4);
        ParentNDerivative=zeros(2,4);        
        for I=1:4
            XPoint = ParentNodes(1,I);
            YPoint = ParentNodes(2,I);
            ShapePart = [1+(GaussPoint(1)-1)/2*XPoint 1+(GaussPoint(2)-1)/2*YPoint];
            wN(I)=0.25*ShapePart(1)*ShapePart(2);
            ParentNDerivative(1,I) = 0.125*XPoint*ShapePart(2);
            ParentNDerivative(2,I) = 0.125*YPoint*ShapePart(1);
        end
    case 11
        wN=zeros(1,4);
        ParentNDerivative=zeros(2,4);        
        for I=1:4
            XPoint = ParentNodes(1,I);
            YPoint = ParentNodes(2,I);
            ShapePart = [1+(GaussPoint(1)+1)/2*XPoint 1+(GaussPoint(2)-1)/2*YPoint];
            wN(I)=0.25*ShapePart(1)*ShapePart(2);
            ParentNDerivative(1,I) = 0.125*XPoint*ShapePart(2);
            ParentNDerivative(2,I) = 0.125*YPoint*ShapePart(1);
        end
    case 12
        wN=zeros(1,8);
        ParentNDerivative=zeros(2,8);
        for I=1:4
            XPoint = ParentNodes(1,I);
            YPoint = ParentNodes(2,I);
            ShapePart = [1+(GaussPoint(1)-1)/2*XPoint 1+(GaussPoint(2)-1)/2*YPoint 1+(GaussPoint(2)+1)/2*YPoint];
            wN(I)=0.25*ShapePart(1)*ShapePart(2)*(1+GaussPoint(2))/2;
            wN(I+4)=0.25*ShapePart(1)*ShapePart(3)*(1-GaussPoint(2))/2;
            ParentNDerivative(1,I) = 0.125*XPoint*ShapePart(2)*(1+GaussPoint(2))/2;
            ParentNDerivative(2,I) = 0.125*YPoint*ShapePart(1)*(1+GaussPoint(2))/2 + 0.125*ShapePart(1)*ShapePart(2);
            ParentNDerivative(1,I+4) = 0.125*XPoint*ShapePart(3)*(1-GaussPoint(2))/2;
            ParentNDerivative(2,I+4) = 0.125*YPoint*ShapePart(1)*(1-GaussPoint(2))/2 - 0.125*ShapePart(1)*ShapePart(3);
        end
    case 13
        wN=zeros(1,8);
        ParentNDerivative=zeros(2,8);
        for I=1:4
            XPoint = ParentNodes(1,I);
            YPoint = ParentNodes(2,I);
            ShapePart = [1+(GaussPoint(1)+1)/2*XPoint 1+(GaussPoint(1)-1)/2*XPoint 1+(GaussPoint(2)+1)/2*YPoint];
            wN(I)=0.25*ShapePart(1)*ShapePart(3)*(1-GaussPoint(1))/2;
            wN(I+4)=0.25*ShapePart(2)*ShapePart(3)*(1+GaussPoint(1))/2;
            ParentNDerivative(1,I) = 0.125*XPoint*ShapePart(3)*(1-GaussPoint(1))/2 - 0.125*ShapePart(1)*ShapePart(3);
            ParentNDerivative(2,I) = 0.125*YPoint*ShapePart(1)*(1-GaussPoint(1))/2;
            ParentNDerivative(1,I+4) = 0.125*XPoint*ShapePart(3)*(1+GaussPoint(1))/2 + 0.125*ShapePart(2)*ShapePart(3);
            ParentNDerivative(2,I+4) = 0.125*YPoint*ShapePart(2)*(1+GaussPoint(1))/2;
        end
    case 14
        wN=zeros(1,8);
        ParentNDerivative=zeros(2,8);
        for I=1:4
            XPoint = ParentNodes(1,I);
            YPoint = ParentNodes(2,I);
            ShapePart = [1+(GaussPoint(1)+1)/2*XPoint 1+(GaussPoint(2)-1)/2*YPoint 1+(GaussPoint(2)+1)/2*YPoint];
            wN(I)=0.25*ShapePart(1)*ShapePart(2)*(1+GaussPoint(2))/2;
            wN(I+4)=0.25*ShapePart(1)*ShapePart(3)*(1-GaussPoint(2))/2;
            ParentNDerivative(1,I) = 0.125*XPoint*ShapePart(2)*(1+GaussPoint(2))/2;
            ParentNDerivative(2,I) = 0.125*YPoint*ShapePart(1)*(1+GaussPoint(2))/2 + 0.125*ShapePart(1)*ShapePart(2);
            ParentNDerivative(1,I+4) = 0.125*XPoint*ShapePart(3)*(1-GaussPoint(2))/2;
            ParentNDerivative(2,I+4) = 0.125*YPoint*ShapePart(1)*(1-GaussPoint(2))/2 - 0.125*ShapePart(1)*ShapePart(3);
        end
    case 15
        wN=zeros(1,8);
        ParentNDerivative=zeros(2,8);
        for I=1:4
            XPoint = ParentNodes(1,I);
            YPoint = ParentNodes(2,I);
            ShapePart = [1+(GaussPoint(1)+1)/2*XPoint 1+(GaussPoint(1)-1)/2*XPoint 1+(GaussPoint(2)-1)/2*YPoint];
            wN(I)=0.25*ShapePart(1)*ShapePart(3)*(1-GaussPoint(1))/2;
            wN(I+4)=0.25*ShapePart(2)*ShapePart(3)*(1+GaussPoint(1))/2;
            ParentNDerivative(1,I) = 0.125*XPoint*ShapePart(3)*(1-GaussPoint(1))/2 - 0.125*ShapePart(1)*ShapePart(3);
            ParentNDerivative(2,I) = 0.125*YPoint*ShapePart(1)*(1-GaussPoint(1))/2;
            ParentNDerivative(1,I+4) = 0.125*XPoint*ShapePart(3)*(1+GaussPoint(1))/2 + 0.125*ShapePart(2)*ShapePart(3);
            ParentNDerivative(2,I+4) = 0.125*YPoint*ShapePart(2)*(1+GaussPoint(1))/2;
        end
    case 16
        wN=zeros(1,16);
        ParentNDerivative=zeros(2,16);
        for I=1:4
            XPoint = ParentNodes(1,I);
            YPoint = ParentNodes(2,I);
            ShapePart = [1+(GaussPoint(1)+1)/2*XPoint 1+(GaussPoint(1)-1)/2*XPoint ...
                1+(GaussPoint(2)+1)/2*YPoint 1+(GaussPoint(2)-1)/2*YPoint];
            wN(I)=0.25*ShapePart(1)*ShapePart(4)*(1-GaussPoint(1))*(1+GaussPoint(2))/4;
            wN(I+4)=0.25*ShapePart(1)*ShapePart(3)*(1-GaussPoint(1))*(1-GaussPoint(2))/4;
            wN(I+8)=0.25*ShapePart(2)*ShapePart(4)*(1+GaussPoint(1))*(1+GaussPoint(2))/4;
            wN(I+12)=0.25*ShapePart(2)*ShapePart(3)*(1+GaussPoint(1))*(1-GaussPoint(2))/4;           
            ParentNDerivative(1,I) = 0.125*XPoint*ShapePart(4)*(1-GaussPoint(1))*(1+GaussPoint(2))/4 ...
                - 0.25*ShapePart(1)*ShapePart(4)*(1+GaussPoint(2))/4;
            ParentNDerivative(2,I) = 0.125*YPoint*ShapePart(1)*(1-GaussPoint(1))*(1+GaussPoint(2))/4 ...
                + 0.25*ShapePart(1)*ShapePart(4)*(1-GaussPoint(1))/4;
            ParentNDerivative(1,I+4) = 0.125*XPoint*ShapePart(3)*(1-GaussPoint(1))*(1-GaussPoint(2))/4 ...
                - 0.25*ShapePart(1)*ShapePart(3)*(1-GaussPoint(2))/4;
            ParentNDerivative(2,I+4) = 0.125*YPoint*ShapePart(1)*(1-GaussPoint(1))*(1-GaussPoint(2))/4 ...
                - 0.25*ShapePart(1)*ShapePart(3)*(1-GaussPoint(1))/4;
            ParentNDerivative(1,I+8) = 0.125*XPoint*ShapePart(4)*(1+GaussPoint(1))*(1+GaussPoint(2))/4 ...
                + 0.25*ShapePart(2)*ShapePart(4)*(1+GaussPoint(2))/4;
            ParentNDerivative(2,I+8) = 0.125*YPoint*ShapePart(2)*(1+GaussPoint(1))*(1+GaussPoint(2))/4 ...
                + 0.25*ShapePart(2)*ShapePart(4)*(1+GaussPoint(1))/4;
            ParentNDerivative(1,I+12) = 0.125*XPoint*ShapePart(3)*(1+GaussPoint(1))*(1-GaussPoint(2))/4 ...
                + 0.25*ShapePart(2)*ShapePart(3)*(1-GaussPoint(2))/4;
            ParentNDerivative(2,I+12) = 0.125*YPoint*ShapePart(2)*(1+GaussPoint(1))*(1-GaussPoint(2))/4 ...
                - 0.25*ShapePart(2)*ShapePart(3)*(1+GaussPoint(1))/4;            
        end        
end

dwNdx=Jacobi\ParentNDerivative;% Calculate the derivation of shape function
end