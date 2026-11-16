%% 这个函数用于统一求解各类单元刚度阵，输出是一个元胞数组,输入是本构矩阵(一般杨氏模量为1，泊松比可变)
function eK_set = Ele_stiff_mat(D)
eK_set = cell(1,16);
nip = 3;                                                                   % 为方便，这里统一取各类积分域实现精确积分的最多高斯点数
[GaussCoordinate,GaussWeight] = guassInt(nip);                             % 积分点和系数
for type=1:16
    for LX=1:nip                                                           % 遍历所有高斯点(这里要优化掉for循环，其实用for循环也可以，毕竟每类单刚只计算一次)
        for LY=1:nip
            E1=GaussCoordinate(LX); E2=GaussCoordinate(LY);
            % ElementNodeCoordinate=[-1 -1;1 -1;1 1;-1 1];
            % 因为所有离散积分域均为相同的正方形区域，因此单元节点坐标可统一定为[-1 -1;1 -1;1 1;-1
            % 1],这样便无需考虑坐标变换,即jacobi为单位阵
            % [~,~, JacobiDET,Jacobi] = ShapeFunction([E1 E2], ElementNodeCoordinate);
            Jacobi=eye(2); JacobiDET=1;
            [wN,dwNdx] = Shape_weightFunction([E1 E2], Jacobi, type);
            FAC=GaussWeight(LX)*GaussWeight(LY)*JacobiDET;
            B = zeros(3,2*size(wN,2));
            for I=1:size(wN,2)
                COL=(I-1)*2+1:(I-1)*2+2;
                B(:,COL)=[dwNdx(1,I) 0; 0 dwNdx(2,I); dwNdx(2,I) dwNdx(1,I)];
            end
            EK = B'*D*B;
            if isempty(eK_set{type})
                eK_set{type} = FAC*EK;
            else
                eK_set{type}=eK_set{type}+FAC*EK;
            end
        end
    end
end
end

