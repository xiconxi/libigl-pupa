//
// Created by pupa on 2021/1/22.
//
#include <igl/opengl/glfw/Viewer.h>
#include <igl/harmonic.h>
#include <igl/diag.h>
#include <igl/min_quad_with_fixed.h>
#include <igl/per_vertex_normals.h>
#include <igl/triangle/triangulate.h>

#include <igl/png/writePNG.h>


void fiber_triangulation(Eigen::MatrixX3d& V, Eigen::MatrixX3i& F, Eigen::VectorXi& b, size_t k = 80) {
    Eigen::MatrixXd V0(k, 2), H(1, 2), V1, V2;
    Eigen::MatrixXi E1(k, 2), E2(k, 2), F1, F2;
    for(size_t i = 0; i < k; i++) {
        V0.row(i) << std::cos(double(i)/k*2*M_PI) ,  std::sin(double(i)/k*2*M_PI) ;
        E1.row(i) << i, (i+1)%k;
        E2.row(i) << (i+1)%k, i;
    }
    H << 10, 10;

    igl::triangle::triangulate(V0, E1, H, "a0.005q32.5",V1,F1);
    igl::triangle::triangulate(V0, E2, H, "a0.005q32.5",V2,F2);

    b = igl::colon<int>(0, k-1);
    V = Eigen::MatrixX3d::Ones(V1.rows()+V2.rows()-k, 3)*0.001;
    F.resize(F1.rows()+F2.rows(), 3);
    V.topLeftCorner(V1.rows(), 2) = V1;
    V.topRightCorner(V1.rows(), 1) *= -1;

    V.bottomLeftCorner(V2.rows()-V0.rows(), 2) = V2.bottomRows(V2.rows()-V0.rows());

    F.topRows(F1.rows()) = F1;
    F.bottomRows(F2.rows()) = F2;
    for(size_t i = F1.rows(); i < F.rows(); i++) {
        F(i, 0) += F(i, 0) < k ? 0:V1.rows()-k;
        F(i, 1) += F(i, 1)< k ? 0:V1.rows()-k;
        F(i, 2) += F(i, 2)< k ? 0:V1.rows()-k;
    }
    for(size_t i = 0; i < F1.rows(); i++)
        std::swap(F(i,0), F(i,1));
}


void uniform_laplacian(Eigen::MatrixX3i& F, Eigen::SparseMatrix<double>& L) {
    Eigen::SparseMatrix<double> A, Adiag;
    Eigen::SparseVector<double> Asum;
    igl::adjacency_matrix(F,A);
    igl::sum(A, 1, Asum);
    igl::diag(Asum, Adiag);
    L = (A - Adiag);
}

void intrinsic(Eigen::VectorXd& e, Eigen::MatrixX3i& F, Eigen::MatrixX3d& E ) {
    E.resize(F.rows(), 3);
    for(size_t i = F.rows(); i < F.rows(); i++) {
        size_t v0 = F(i, 0), v1 = F(i, 1), v2 = F(i, 2);
        E(i, 0) = (e[v1] + e[v2]) / 2;
        E(i, 1) = (e[v2] + e[v0]) / 2;
        E(i, 2) = (e[v0] + e[v1]) / 2;
    }
}

void cotmatrix(Eigen::VectorXd& e, Eigen::MatrixX3i& F, Eigen::SparseMatrix<double>& L) {
    Eigen::MatrixX3d E(F.rows(), 3);
    intrinsic(e, F, E);
    Eigen::MatrixXd  C;
    igl::cotmatrix_entries(E, C);

    Eigen::MatrixX2i edges(3, 2);
    edges << 1,2,  2,0, 0,1;
    std::vector<Eigen::Triplet<double> > IJV;
    IJV.reserve(F.rows()*6);
    for(int i = 0; i < F.rows(); i++)
    {
        for(int e = 0; e < edges.rows();e++)
        {
            int source = F(i,edges(e,0));
            int dest = F(i,edges(e,1));
            IJV.push_back({source,dest,C(i,e)});
            IJV.push_back({dest,source,C(i,e)});
            IJV.push_back({source,source,-C(i,e)});
            IJV.push_back({dest,dest,-C(i,e)});
        }
    }
    L.resize(e.rows(), e.rows());
    L.setFromTriplets(IJV.begin(),IJV.end());
}

Eigen::VectorXd vertex_edge_length(Eigen::MatrixX3d& V, Eigen::MatrixX3i& F) {
    Eigen::MatrixX3d E;
    Eigen::MatrixX2d e = Eigen::MatrixX2d::Zero(V.rows(), 2);
    igl::edge_lengths(V, F, E);
    for(size_t i = 0; i < F.rows(); i++) {
        e.row(F(i, 0)) += Eigen::RowVector2d(E(i, 1), 1);
        e.row(F(i, 0)) += Eigen::RowVector2d(E(i, 2), 1);
        e.row(F(i, 1)) += Eigen::RowVector2d(E(i, 0), 1);
        e.row(F(i, 1)) += Eigen::RowVector2d(E(i, 2), 1);
        e.row(F(i, 2)) += Eigen::RowVector2d(E(i, 0), 1);
        e.row(F(i, 2)) += Eigen::RowVector2d(E(i, 1), 1);
    }
    return (e.col(0).array()/e.col(1).array()).matrix();
}

class FiberMesh {
public:
    FiberMesh(Eigen::MatrixX3d& V_, Eigen::MatrixX3i& F, Eigen::VectorXi& b): V(V_), F_(F), b_(b){
        // Edge vector Matrix
        std::vector<Eigen::Triplet<double> > triplets;
        triplets.reserve(F_.rows() * 6);
        for (size_t i = 0; i < F_.rows(); i++) {
            size_t v0 = F_(i, 0), v1 = F_(i, 1), v2 = F_(i, 2);
            triplets.push_back({i*3+0, v1, -1});
            triplets.push_back({i*3+0, v2, 1});
            triplets.push_back({i*3+1, v2, -1});
            triplets.push_back({i*3+1, v0, 1});
            triplets.push_back({i*3+2, v0, -1});
            triplets.push_back({i*3+2, v1, 1});
        }
        E.resize(F_.rows()* 3, V.rows());
        E.setFromTriplets(triplets.begin(), triplets.end());

        igl::speye(V.rows(), I);

        uniform_laplacian(F_, L_u);
    }

    
    void optimize_normal(int n) {
        igl::per_vertex_normals(V, F_, igl::PER_VERTEX_NORMALS_WEIGHTING_TYPE_AREA, N);

        uniform_solver_.compute(L_u.transpose() * L_u + I);
        for (int smooth_i = 0; smooth_i < n; smooth_i++) {
            N = uniform_solver_.solve(N).rowwise().normalized();
        }
        uniform_solver_.compute(L_u.transpose() * L_u + I);

        e = vertex_edge_length(V, F_);
        for (int smooth_i = 0; smooth_i < n; smooth_i++) {
            e = uniform_solver_.solve(e);
        }
        cotmatrix(e, F_, L);

        igl::massmatrix(V, F_, igl::MASSMATRIX_TYPE_DEFAULT, M);
        igl::invert_diag(M, Minv);
        H = Minv * (L * V).rowwise().norm();
        H.setConstant(uniform_solver_.solve(H).mean() );
    }

    void solve() {
        igl::massmatrix(V, F_, igl::MASSMATRIX_TYPE_DEFAULT, M);
        igl::invert_diag(M, Minv);
        H = Minv * (L * V).rowwise().norm();
//        H.setConstant((L_u * V).rowwise().norm().mean() );
        e = uniform_solver_.solve(vertex_edge_length(V, F_));

        N = uniform_solver_.solve(N).rowwise().normalized();
        Eigen::MatrixX3d delta = -1 * H.asDiagonal() * N;
        igl::massmatrix(V, F_, igl::MASSMATRIX_TYPE_DEFAULT, M);

        // re-scale edge vector(riemann metric ??)
        Eigen::MatrixX3d eta = (E * V).rowwise().normalized();
        for (size_t fi = 0; fi < F_.rows(); fi++) {
            size_t v0 = F_(fi, 0), v1 = F_(fi, 1), v2 = F_(fi, 2);
            eta.row(fi * 3 + 0) *= (e[v1] + e[v2]) / 2;
            eta.row(fi * 3 + 1) *= (e[v2] + e[v0]) / 2;
            eta.row(fi * 3 + 2) *= (e[v0] + e[v1]) / 2;
        }

        // argmin_v { || Lv - \Delta||^ 2 + || Ev-\eta || ^ 2}, s.t. planar curve
        igl::min_quad_with_fixed_data<double> quad_data;
        Eigen::SparseMatrix<double> Q = ( L.transpose() * L + E.transpose() * E * 0.0  + M );
        Eigen::SparseMatrix<double> Aeq = L.topRows(b_.size());
        Eigen::MatrixX3d B = -( L.transpose() *  delta + E.transpose() * eta * 0.0 + M * V);
        min_quad_with_fixed_precompute(Q, b_, Aeq, true, quad_data);
        assert(igl::min_quad_with_fixed_solve(quad_data, B, V.topRows(b_.size()), delta.topRows(b_.size()), V));
    }

    void optimize_vertices(int n = 5) {
        if(n == 0) {
            igl::massmatrix(V, F_, igl::MASSMATRIX_TYPE_DEFAULT, M);
            igl::invert_diag(M, Minv);
            H = Minv * (L * V).rowwise().norm();
            H.setConstant(uniform_solver_.solve(H).mean() );
            cotmatrix(e, F_, L);
            return ;
        }


        // re-scale laplace-coordinate
        N = uniform_solver_.solve(N).rowwise().normalized();
        Eigen::MatrixX3d delta = -1 * M * H.asDiagonal() * N;

        // re-scale edge vector(riemann metric ??)
        Eigen::MatrixX3d eta = (E * V).rowwise().normalized();
        e = uniform_solver_.solve(vertex_edge_length(V, F_));
        for (size_t fi = 0; fi < F_.rows(); fi++) {
            size_t v0 = F_(fi, 0), v1 = F_(fi, 1), v2 = F_(fi, 2);
            eta.row(fi * 3 + 0) *= (e[v1] + e[v2]) / 2;
            eta.row(fi * 3 + 1) *= (e[v2] + e[v0]) / 2;
            eta.row(fi * 3 + 2) *= (e[v0] + e[v1]) / 2;
        }

        // argmin_v { || Lv - \Delta||^ 2 + || Ev-\eta || ^ 2}, s.t. planar curve
        igl::min_quad_with_fixed_data<double> quad_data;
        Eigen::SparseMatrix<double> Q = ( L.transpose()  * L  + E.transpose() * E * 0.5  );
        Eigen::MatrixX3d B = -( L.transpose() * delta + E.transpose() * eta * 0.5  );
        min_quad_with_fixed_precompute(Q, b_, Eigen::SparseMatrix<double>(), true, quad_data);
        assert(igl::min_quad_with_fixed_solve(quad_data, B, V.topRows(b_.size()), Eigen::VectorXd(), V));

        optimize_vertices(n-1);
    }

    void smoothing(double lambda) {
        Eigen::SparseMatrix<double> _L;
        igl::cotmatrix(V, F_, _L);
        igl::min_quad_with_fixed_data<double> quad_data;
        Eigen::SparseMatrix<double> Q = ( _L.transpose() *_L + lambda * I   );
        Eigen::MatrixX3d B = - lambda * V;
        min_quad_with_fixed_precompute(Q, b_, Eigen::SparseMatrix<double>(), true, quad_data);
        assert(igl::min_quad_with_fixed_solve(quad_data, B, V.topRows(b_.size()), Eigen::VectorXd(), V));
    }

    void shape_optimization(double lambda) {
        Eigen::SparseMatrix<double> _L;
        igl::cotmatrix(V, F_, _L);
        igl::min_quad_with_fixed_data<double> quad_data;
        Eigen::SparseMatrix<double> Q = ( L.transpose()  * L + lambda * I);
        Eigen::MatrixX3d B = - (lambda * I + L.transpose() * _L ) * V ;
        min_quad_with_fixed_precompute(Q, b_, Eigen::SparseMatrix<double>(), true, quad_data);
        assert(igl::min_quad_with_fixed_solve(quad_data, B, V.topRows(b_.size()), Eigen::VectorXd(), V));
    }

    Eigen::MatrixX3d V, N;
    Eigen::VectorXd H, e;
private:
    Eigen::SparseMatrix<double> L, L_u, I, E, M, Minv;
    Eigen::SparseLU<Eigen::SparseMatrix<double> > uniform_solver_;
    
    Eigen::MatrixX3i& F_;
    Eigen::VectorXi& b_;

};



int main(int argc, char** argv) {
    igl::opengl::glfw::Viewer viewer;
    viewer.resize(800, 600);

    Eigen::MatrixX3d V, N;
    Eigen::MatrixX3i F;
    Eigen::VectorXi b;
    fiber_triangulation(V, F, b, 60);


    FiberMesh fiber_mesh(V, F, b);
    fiber_mesh.optimize_normal(80);

    viewer.data().set_mesh(fiber_mesh.V, F);


    Eigen::Matrix<std::uint8_t,-1,-1> R(800,600),G(800,600),B(800,600),A(800,600);
    viewer.callback_key_pressed = [&](igl::opengl::glfw::Viewer &viewer,std::uint8_t key,int mod)->bool{
        if (key == ' ') {
            fiber_mesh.optimize_vertices(5);
            viewer.data().set_mesh(fiber_mesh.V, F);
            igl::per_vertex_normals(fiber_mesh.V, F, N);
            viewer.data().set_normals(N);
            static size_t frame_cnt = 0;
            viewer.core().draw_buffer(viewer.data(), false, R, G, B, A);
            igl::png::writePNG(R, G, B, A, "out" + std::to_string(frame_cnt++) + ".png");
        }else if (key == '2') {
            fiber_mesh.solve();
            viewer.data().set_mesh(fiber_mesh.V, F);
//            viewer.core().draw_buffer(viewer.data(), false, R, G, B, A);
//            igl::png::writePNG(R, G, B, A, "out" + std::to_string(frame_cnt++) + ".png");
        }else if (key == 'n') {
            viewer.data().set_mesh(fiber_mesh.N, F);
        }else if (key == 'v') {
            viewer.data().set_mesh(fiber_mesh.V, F);
        }else if (key == 's') {
            fiber_mesh.smoothing(0.5);
            viewer.data().set_mesh(fiber_mesh.V, F);
        }else if (key == 'x') {
            fiber_mesh.shape_optimization(1);
            viewer.data().set_mesh(fiber_mesh.V, F);
        }
        return false;
    };


    return viewer.launch();

}
